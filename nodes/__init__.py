# nodes/__init__.py
# Agrega os mappings dos tres nodes do pack.
#
# CRITICO: imports RELATIVOS obrigatorios. Um `from nodes import ...` absoluto
# resolviria para o nodes.py do core do ComfyUI (que esta no sys.path),
# carregando o pack sem registrar nenhum node — silenciosamente. Bug da v0.1.0.
from .gemini_image_interactions import (
    NODE_CLASS_MAPPINGS as _GEMINI_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _GEMINI_DISPLAY,
)
from .aw_gpt_image import (
    NODE_CLASS_MAPPINGS as _GPT_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _GPT_DISPLAY,
)
from .aw_kling_video import (
    NODE_CLASS_MAPPINGS as _KLING_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _KLING_DISPLAY,
)

NODE_CLASS_MAPPINGS = {**_GEMINI_CLASS, **_GPT_CLASS, **_KLING_CLASS}
NODE_DISPLAY_NAME_MAPPINGS = {**_GEMINI_DISPLAY, **_GPT_DISPLAY, **_KLING_DISPLAY}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
