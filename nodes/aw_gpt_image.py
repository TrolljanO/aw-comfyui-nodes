"""
AwGptImageNode — ComfyUI custom node for OpenAI GPT-Image API (edits + generations).

WHY THIS NODE EXISTS
--------------------
The upstream GPTImageNode from ComfyUI-ExternalAPI-Helpers (Aryan185) declares
`model` and `size` as COMBO inputs. ComfyUI validates COMBO inputs server-side
and returns "value_not_in_list" for any id that is not in the hardcoded list.
This blocks every model released after the node was authored — concretely,
gpt-image-2.5-flare and gpt-image-2.5-sunburst cannot be selected, and new
size options (2048×2048, etc.) are equally locked out.

This node replaces both COMBO fields with STRING so any current or future model
id works without a node update.  Everything else follows the upstream behaviour
faithfully.

REST surface:
  edits:       POST https://api.openai.com/v1/images/edits  (multipart/form-data)
  generations: POST https://api.openai.com/v1/images/generations (application/json)
Auth: Authorization: Bearer <key>
"""

import base64
import io
import os

import requests

# Optional .env loader for local dev; ignored in production ComfyUI deployments.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

# Heavy deps (numpy, torch, PIL) are imported lazily inside functions so the
# module loads cleanly in environments where they are absent (lint / type checks).

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EDITS_ENDPOINT = "https://api.openai.com/v1/images/edits"
_GENERATIONS_ENDPOINT = "https://api.openai.com/v1/images/generations"
REQUEST_TIMEOUT = 180  # seconds — image generation can be slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_api_key(raw_value: str) -> str:
    """
    Pack convention for api_key inputs:
    - If the string is all-uppercase with no spaces (looks like an env-var name),
      resolve via os.environ.
    - Otherwise treat it as a literal key value.
    """
    stripped = raw_value.strip()
    if stripped and stripped == stripped.upper() and " " not in stripped:
        env_val = os.environ.get(stripped, "")
        if env_val:
            return env_val
    return stripped


def _tensor_to_png_bytes(tensor) -> bytes:
    """
    Convert a ComfyUI IMAGE tensor ([H,W,3] or [1,H,W,3], float32 0-1) to
    raw PNG bytes.
    """
    import numpy as np      # noqa: PLC0415 — always present in ComfyUI
    from PIL import Image   # noqa: PLC0415

    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = (tensor.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_tensor_to_png_bytes(mask) -> bytes:
    """
    Convert a ComfyUI MASK tensor ([H,W] or [1,H,W], float32 0-1) to a PNG with
    transparency channel.  The OpenAI edits API uses the alpha channel of the mask
    to indicate which areas to regenerate: transparent pixels are regenerated,
    opaque pixels are preserved.  ComfyUI masks are 0=masked/transparent,
    1=unmasked/opaque, so we invert the mask value into the alpha channel.
    """
    import numpy as np      # noqa: PLC0415
    from PIL import Image   # noqa: PLC0415

    # Normalise shape to [H, W]
    if mask.ndim == 3:
        mask = mask[0]
    arr = (mask.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)

    mask_l = Image.fromarray(arr, mode='L')
    # Invert: ComfyUI mask=0 → transparent (to regenerate) → alpha=0
    inverted_alpha = Image.eval(mask_l, lambda x: 255 - x)
    rgba = Image.new("RGBA", mask_l.size)
    rgba.putalpha(inverted_alpha)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


def _b64_to_tensor(b64_data: str):
    """
    Decode a base64 image string to a ComfyUI IMAGE tensor [1, H, W, 3] float32.
    """
    import numpy as np      # noqa: PLC0415
    import torch            # noqa: PLC0415
    from PIL import Image   # noqa: PLC0415

    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class AwGptImageNode:
    """
    ComfyUI node that calls the OpenAI GPT-Image API to generate or edit images.

    When one or more image_N inputs are connected the node calls the edits
    endpoint (multipart form-data).  Otherwise it calls the generations endpoint
    (JSON).  In both cases the response's data[].b64_json fields are decoded and
    batched into a single IMAGE tensor.
    """

    CATEGORY = "aw/openai"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A photorealistic architectural render",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "OPENAI_API_KEY",
                    },
                ),
                # STRING (not COMBO) — reason this node exists.
                # ComfyUI validates COMBO values server-side and rejects any id
                # not in the compiled list, blocking gpt-image-2.5-* models.
                "model": (
                    "STRING",
                    {
                        "default": "gpt-image-2",
                    },
                ),
                "background": (
                    ["auto", "transparent", "opaque"],
                    {
                        "default": "auto",
                    },
                ),
                # STRING (not COMBO) — same rationale as model: the API accepts
                # sizes beyond the three values the upstream node hardcoded.
                "size": (
                    "STRING",
                    {
                        "default": "1024x1024",
                    },
                ),
                # Extended quality values: the upstream COMBO had only 4; the API
                # now documents xhigh and max for some models.
                "quality": (
                    ["auto", "low", "medium", "high", "xhigh", "max"],
                    {
                        "default": "auto",
                    },
                ),
                "output_format": (
                    ["png", "jpeg", "webp"],
                    {
                        "default": "png",
                    },
                ),
                "output_compression": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "n_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                    },
                ),
                # seed: the OpenAI images API does not support a seed parameter.
                # Accepted here for compatibility with the WorkflowParameterMapper
                # which injects seed into all nodes — the value is silently ignored.
                # See the GeminiImageInteractionsNode for the same pattern.
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        api_key: str,
        model: str,
        background: str,
        size: str,
        quality: str,
        output_format: str,
        output_compression: int,
        n_images: int,
        seed: int,  # noqa: ARG002 — intentionally unused; see INPUT_TYPES comment
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        mask=None,
    ):
        import torch  # noqa: PLC0415 — always present in ComfyUI

        # ---- resolve key ---------------------------------------------------
        resolved_key = _resolve_api_key(api_key)
        if not resolved_key:
            raise RuntimeError(
                "AwGptImageNode: api_key resolved to an empty string. "
                "Set OPENAI_API_KEY in the ComfyUI .env or provide the key directly."
            )

        headers = {"Authorization": f"Bearer {resolved_key}"}

        # ---- collect input images ------------------------------------------
        input_images = [
            img for img in (image_1, image_2, image_3, image_4, image_5)
            if img is not None
        ]
        is_edit = len(input_images) > 0

        # ---- common payload fields -----------------------------------------
        payload = {
            "model": model.strip(),
            "prompt": prompt,
            "background": background,
            "n": n_images,
            "size": size.strip(),
            "quality": quality,
            "output_format": output_format,
            "output_compression": output_compression,
        }

        # ---- dispatch ----------------------------------------------------------
        if is_edit:
            # Multipart form-data: one 'image[]' field per input image
            files = [
                ("image[]", (f"input_{i}.png", _tensor_to_png_bytes(img), "image/png"))
                for i, img in enumerate(input_images)
            ]
            if mask is not None:
                files.append(("mask", ("mask.png", _mask_tensor_to_png_bytes(mask), "image/png")))

            response = requests.post(
                _EDITS_ENDPOINT,
                headers=headers,
                data=payload,   # form fields alongside the file parts
                files=files,
                timeout=REQUEST_TIMEOUT,
            )
        else:
            # JSON body for generations (no images)
            response = requests.post(
                _GENERATIONS_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

        # ---- error handling ------------------------------------------------
        if response.status_code != 200:
            snippet = response.text[:300]
            raise RuntimeError(
                f"AwGptImageNode: API returned HTTP {response.status_code}. "
                f"Body (truncated to 300 chars): {snippet}"
            )

        # ---- decode response -----------------------------------------------
        result = response.json()
        data_items = result.get("data", [])
        if not data_items:
            raise RuntimeError(
                "AwGptImageNode: API response contained no data items. "
                f"Full response keys: {list(result.keys())}"
            )

        output_tensors = []
        for item in data_items:
            b64 = item.get("b64_json")
            if not b64:
                raise RuntimeError(
                    "AwGptImageNode: A data item contained no b64_json field. "
                    f"Item keys: {list(item.keys())}"
                )
            output_tensors.append(_b64_to_tensor(b64))  # [1, H, W, 3]

        # Batch all images into a single tensor [N, H, W, 3]
        try:
            batched = torch.cat(output_tensors, dim=0)
        except RuntimeError:
            sizes = [tuple(t.shape[1:]) for t in output_tensors]
            raise RuntimeError(
                f"AwGptImageNode: Cannot batch {n_images} images — API returned "
                f"mixed sizes: {sizes}. Set 'size' to a fixed value (e.g. 1024x1024) "
                f"instead of 'auto' to avoid this."
            )

        return (batched,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AwGptImageNode": AwGptImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AwGptImageNode": "GPT Image (aw)",
}
