from tqdm.auto import tqdm, trange
import os
import time
import json
import math
import torch
import ujson
import textwrap
import folder_paths
import numpy as np
from typing import Any, Dict, Iterable, List, Optional
from torchvision.transforms import ToPILImage
from boltons.iterutils import remap
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
)
import comfy.model_management
from pathlib import Path
# from diffusers import BriaFiboEditPipeline
from .pipeline_bria_fibo_edit import BriaFiboEditPipeline
from diffusers.quantizers import PipelineQuantizationConfig

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def clean_json(caption):
    caption["pickascore"] = 1.0
    caption["aesthetic_score"] = 10.0
    caption = prepare_clean_caption(caption)
    return caption

def is_valid_edit_json(json_input: str | dict):
    """
    Check if the input is a valid JSON string or dict with an "edit_instruction" key.

    Args:
        json_input (`str` or `dict`):
            The JSON string or dict to check.

    Returns:
        `bool`: True if the input is a valid JSON string or dict with an "edit_instruction" key, False otherwise.
    """
    try:
        if isinstance(json_input, str) and "edit_instruction" in json_input:
            json.loads(json_input)
            return True
        elif isinstance(json_input, dict) and "edit_instruction" in json_input:
            return True
        else:
            return False
    except json.JSONDecodeError:
        return False

def parse_aesthetic_score(record: dict) -> str:
    ae = record["aesthetic_score"]
    if ae < 5.5:
        return "very low"
    elif ae < 6:
        return "low"
    elif ae < 7:
        return "medium"
    elif ae < 7.6:
        return "high"
    else:
        return "very high"

def parse_pickascore(record: dict) -> str:
    ps = record["pickascore"]
    if ps < 0.78:
        return "very low"
    elif ps < 0.82:
        return "low"
    elif ps < 0.87:
        return "medium"
    elif ps < 0.91:
        return "high"
    else:
        return "very high"

def prepare_clean_caption(record: dict) -> str:
    def keep(p, k, v):
        is_none = v is None
        is_empty_string = isinstance(v, str) and v == ""
        is_empty_dict = isinstance(v, dict) and not v
        is_empty_list = isinstance(v, list) and not v
        is_nan = isinstance(v, float) and math.isnan(v)
        if is_none or is_empty_string or is_empty_list or is_empty_dict or is_nan:
            return False
        return True

    try:
        scores = {}
        if "pickascore" in record:
            scores["preference_score"] = parse_pickascore(record)
        if "aesthetic_score" in record:
            scores["aesthetic_score"] = parse_aesthetic_score(record)

        # Create structured caption dict of original values
        fields = [
            "short_description",
            "objects",
            "background_setting",
            "lighting",
            "aesthetics",
            "photographic_characteristics",
            "style_medium",
            "text_render",
            "context",
            "artistic_style",
        ]

        original_caption_dict = {f: record[f] for f in fields if f in record}

        # filter empty values recursivly (i.e. None, "", {}, [], float("nan"))
        clean_caption_dict = remap(original_caption_dict, visit=keep)

        # Set aesthetics scores
        if "aesthetics" not in clean_caption_dict:
            if len(scores) > 0:
                clean_caption_dict["aesthetics"] = scores
        else:
            clean_caption_dict["aesthetics"].update(scores)

        # Dumps clean structured caption as minimal json string (i.e. no newlines\whitespaces seps)
        clean_caption_str = ujson.dumps(clean_caption_dict, escape_forward_slashes=False)
        return clean_caption_str
    except Exception as ex:
        print("Error: ", ex)
        raise ex

def _collect_images(messages: Iterable[Dict[str, Any]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            image_value = item.get("image")
            if isinstance(image_value, Image.Image):
                images.append(image_value)
            else:
                raise ValueError("Expected PIL.Image for image content in messages.")
    return images

def _strip_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> str:
    if not stop_sequences:
        return text.strip()
    cleaned = text
    for stop in stop_sequences:
        if not stop:
            continue
        index = cleaned.find(stop)
        if index >= 0:
            cleaned = cleaned[:index]
    return cleaned.strip()

def generate_json_prompt(
    vlm_processor: AutoModelForCausalLM,
    top_p: float,
    temperature: float,
    max_tokens: int,
    stop: List[str],
    image: Optional[Image.Image] = None,
    prompt: Optional[str] = None,
    structured_prompt: Optional[str] = None,
):
    refine_image = None
    if image is None and structured_prompt is None:
        # only got prompt
        task = "generate"
        editing_instructions = None
    elif image is None and structured_prompt is not None and prompt is not None:
        # got structured prompt and prompt
        task = "refine"
        editing_instructions = prompt
    elif image is not None and structured_prompt is None and prompt is not None:
        # got image and prompt
        task = "refine"
        editing_instructions = prompt
        refine_image = image
    elif image is not None and structured_prompt is None and prompt is None:
        # only got image
        task = "inspire"
        editing_instructions = None
    else:
        raise ValueError("Invalid input. The input should contain at least one of the following: prompt, image, json_prompt")

    messages = build_messages(
        task,
        image=image,
        prompt=prompt,
        refine_image=refine_image,
        structured_prompt=structured_prompt,
        editing_instructions=editing_instructions,
    )

    generated_prompt = vlm_processor.generate(
        messages=messages, top_p=top_p, temperature=temperature, max_tokens=max_tokens, stop=stop
    )
    cleaned_json_data = prepare_clean_caption(generated_prompt)
    return cleaned_json_data

def build_messages(
    task: str,
    *,
    image: Optional[Image.Image] = None,
    refine_image: Optional[Image.Image] = None,
    prompt: Optional[str] = None,
    structured_prompt: Optional[str] = None,
    editing_instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []

    if task == "inspire":
        user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": "<inspire>"})
    elif task == "generate":
        text_value = (prompt or "").strip()
        formatted = f"<generate>\n{text_value}"
        user_content.append({"type": "text", "text": formatted})
    else:  # refine
        if refine_image is None:
            base_prompt = (structured_prompt or "").strip()
            edits = (editing_instructions or "").strip()
            formatted = textwrap.dedent(
                f"""<refine>
Input:
{base_prompt}
Editing instructions:
{edits}"""
            ).strip()
            user_content.append({"type": "text", "text": formatted})
        else:
            user_content.append({"type": "image", "image": refine_image})
            edits = (editing_instructions or "").strip()
            formatted = textwrap.dedent(
                f"""<refine>
Editing instructions:
{edits}"""
            ).strip()
            user_content.append({"type": "text", "text": formatted})

    messages: List[Dict[str, Any]] = []
    messages.append({"role": "user", "content": user_content})
    return messages

class TransformersEngine(torch.nn.Module):
    """Inference wrapper using Hugging Face transformers."""

    def __init__(
        self,
        model: str,
        dtype: str,
        *,
        quantization_config = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super(TransformersEngine, self).__init__()
        default_processor_kwargs: Dict[str, Any] = {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1024 * 28 * 28,
        }
        processor_kwargs = {**default_processor_kwargs, **(processor_kwargs or {})}
        model_kwargs = model_kwargs or {}

        self.processor = AutoProcessor.from_pretrained(model, **processor_kwargs)

        print("[Fibo Edit VLM] Loading model...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            dtype=dtype,
            quantization_config=quantization_config,
            **model_kwargs,
        )
        self.model.eval()

        tokenizer_obj = self.processor.tokenizer
        if tokenizer_obj.pad_token_id is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
        self._pad_token_id = tokenizer_obj.pad_token_id
        eos_token_id = tokenizer_obj.eos_token_id
        if isinstance(eos_token_id, list) and eos_token_id:
            self._eos_token_id = eos_token_id
        elif eos_token_id is not None:
            self._eos_token_id = [eos_token_id]
        else:
            raise ValueError("Tokenizer must define an EOS token for generation.")

    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def device(self) -> torch.device:
        return self.model.device

    def _to_model_device(self, value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        target_device = getattr(self.model, "device", None)
        if target_device is None or target_device.type == "meta":
            return value
        if value.device == target_device:
            return value
        return value.to(target_device)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        tokenizer = self.processor.tokenizer
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_inputs: Dict[str, Any] = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
        }
        images = _collect_images(messages)
        if images:
            processor_inputs["images"] = images
        inputs = self.processor(**processor_inputs)
        inputs = {key: self._to_model_device(value) for key, value in inputs.items()}

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._pad_token_id,
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Processor did not return input_ids; cannot compute new tokens.")
        new_token_ids = generated_ids[:, input_ids.shape[-1] :]
        decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        if not decoded:
            return ""
        text = decoded[0]
        stripped_text = _strip_stop_sequences(text, stop)
        json_prompt = json.loads(stripped_text)
        return json_prompt

class FiboEdit_VLM:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.current_quantization = None,
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0, "max": 1, "step": 0.1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 4096, "min": 128, "max": 8192, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "json_prompt": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "ComfyUI-Fibo-Edit-VLM"

    def inference(
        self,
        quantization,
        temperature,
        top_p,
        max_new_tokens,
        seed,
        keep_model_loaded,
        prompt=None,
        image=None,
        json_prompt=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)
        model_id = "briaai/FIBO-vlm"
        self.model_checkpoint = os.path.join(folder_paths.models_dir, "fibo_vlm")

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )
        
        if (
            self.model is None
            or self.current_quantization != quantization
        ):
            self.current_quantization = quantization
            if self.model is not None:
                del self.model
                self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None
            
            self.model = TransformersEngine(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                quantization_config=quantization_config
            )
            if not quantization_config:
                self.model.model.to("cuda")

        if image is not None:
            image = ToPILImage()(image[0].permute(2, 0, 1))
        
        if prompt.strip() == "":
            prompt = None

        if json_prompt.strip() == "":
            json_prompt = None

        with torch.no_grad():
            result_json_prompt = generate_json_prompt(
                vlm_processor=self.model,
                image=image,
                prompt=prompt,
                structured_prompt=json_prompt,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=["<|im_end|>", "<|end_of_text|>"],
            )

            if prompt:
                result_json_prompt = json.loads(result_json_prompt)
                result_json_prompt["edit_instruction"] = prompt.strip()
                result_json_prompt = json.dumps(result_json_prompt)

            if not keep_model_loaded:
                del self.processor  # release processor memory
                del self.model  # release model memory
                self.processor = None  # set processor to None
                self.model = None  # set model to None
                self.current_model_id = None
                self.current_quantization = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # release GPU memory
                    torch.cuda.ipc_collect()

            return (result_json_prompt,)


class FiboEdit:
    def __init__(self):
        self.model = None
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "boredcoder/Fibo-Edit-8bit"}),
                "image": ("IMAGE",),
                "json_prompt": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1},),
                "cfg": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1},),
                "low_vram": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "ComfyUI-Fibo-Edit"
    DESCRIPTION = "(Down)load and use fibo edit. Setting low_vram=True sets cfg=1 to reduce VRAM usage."

    def sample(
        self,
        model,
        image,
        json_prompt,
        steps,
        cfg,
        low_vram,
        seed,
        mask=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)
        
        if not is_valid_edit_json(json_prompt):
            raise ValueError("Prompt has to be a valid JSON string with \"edit_instruction\" key!")

        if os.path.exists(model):
            self.model_checkpoint = model
        else:
            model_id = model
            self.model_checkpoint = os.path.join(folder_paths.models_dir, "fibo_edit", os.path.basename(model_id))

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        pipe = BriaFiboEditPipeline.from_pretrained(
            self.model_checkpoint,
            torch_dtype=torch.bfloat16,
            transformer=None,
        )

        pipe.enable_model_cpu_offload()

        if mask is not None:
            if (mask == 0).all():
                mask = None
            else:
                mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                mask = ToPILImage()(mask[0].permute(2, 0, 1)).convert("L")
        
        if image is not None:
            image = tensor2pil(image)
        
        if low_vram:
            cfg = 1

        with torch.no_grad():
            result = pipe(
                custom_model_path=self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                image=image,
                mask=mask,
                prompt=json_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
            ).images[0]
            result = pil2tensor(result)

        return (result,)
