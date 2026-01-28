from .nodes import FiboEdit_VLM, FiboEdit

WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FiboEdit": FiboEdit,
    "FiboEdit_VLM": FiboEdit_VLM,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FiboEdit": "Fibo Edit",
    "FiboEdit_VLM": "Fibo Edit VLM",
}
