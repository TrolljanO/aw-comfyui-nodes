# __init__.py
"""
aw-comfyui-nodes — a|w ComfyUI custom nodes pack.
Entry point recognised by ComfyUI-Manager.
"""
# Import RELATIVO obrigatorio: a raiz do ComfyUI esta no sys.path e tem um
# modulo `nodes` proprio (o registro core). Um `from nodes import ...` absoluto
# resolve para o core e o pack registra zero nodes, sem erro no log.
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401

WEB_DIRECTORY = None  # no frontend JS assets

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
