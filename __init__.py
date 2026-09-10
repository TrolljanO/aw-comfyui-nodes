# __init__.py
"""
aw-comfyui-nodes — a|w ComfyUI custom nodes pack.
Entry point recognised by ComfyUI-Manager.
"""
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401

WEB_DIRECTORY = None  # no frontend JS assets

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
