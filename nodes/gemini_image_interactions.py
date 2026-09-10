"""
GeminiImageInteractionsNode — ComfyUI custom node for Google Gemini Interactions API.

REST surface: POST https://generativelanguage.googleapis.com/v1beta/interactions
Auth: x-goog-api-key header
Doc reference: https://ai.google.dev/gemini-api/docs/image-generation
"""

import base64
import io
import os
import warnings

import requests

# Optional: load a .env from the pack root for local dev/testing.
# In a real ComfyUI deployment GEMINI_API_KEY is set in the process env.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

# numpy and PIL are imported lazily inside functions to allow the module to
# load in environments where they are absent (e.g. local compile checks).
# In a real ComfyUI environment both are always present.

# ---------------------------------------------------------------------------
# Constants derived from the Interactions API documentation
# ---------------------------------------------------------------------------

INTERACTIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
REQUEST_TIMEOUT = 120  # seconds

# Documented input-image limits per model (total images in the `input` array).
# Source: capability table at https://ai.google.dev/gemini-api/docs/image-generation
# All current 3.x models accept up to 14 total input images.
_MODEL_IMAGE_LIMITS = {
    "gemini-3-pro-image": 14,
    "gemini-3.1-flash-image": 14,
    "gemini-3.1-flash-lite-image": 14,
    "gemini-2.5-flash-image": 14,
}
_DEFAULT_IMAGE_LIMIT = 14  # fallback for unknown models

# Models that support `generation_config.thinking_level` per doc.
# "With Gemini 3.1 Flash Image, you can control the amount of thinking the
#  model uses." — only 3.1-family explicitly supports this parameter.
_THINKING_LEVEL_MODELS = {
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_api_key(raw_value: str) -> str:
    """
    Resolve an API key input following the pack convention:
    - If the value looks like an environment-variable name (no spaces, no
      special chars typical of actual key strings), try os.environ first.
    - If found in env, return the env value.
    - Otherwise return the literal string as-is (user may have pasted the key).

    This matches the behaviour documented in the NODE_REGISTRY:
    'o campo api_key nos nodes aceita o nome da variável de ambiente como
    string (ex: "GEMINI_API_KEY"), não o valor da chave diretamente.'
    """
    stripped = raw_value.strip()
    # Heuristic: env var names are all uppercase/underscore/digit, no spaces
    if stripped and stripped == stripped.upper() and " " not in stripped:
        env_val = os.environ.get(stripped, "")
        if env_val:
            return env_val
    return stripped


def _tensor_to_png_b64(tensor) -> str:
    """
    Convert a ComfyUI IMAGE tensor (shape [H,W,3] or [1,H,W,3], float32 0-1)
    to a base64-encoded PNG string suitable for the Interactions API input.
    """
    import numpy as np      # noqa: PLC0415 — always present in ComfyUI
    import torch            # noqa: PLC0415
    from PIL import Image   # noqa: PLC0415

    if tensor.ndim == 4:
        tensor = tensor[0]  # take first image from batch
    # Clamp and convert to uint8
    arr = (tensor.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _png_b64_to_tensor(b64_data: str):
    """
    Decode a base64 image (any format) into a ComfyUI IMAGE tensor
    [1, H, W, 3] float32 in [0, 1].
    """
    import numpy as np      # noqa: PLC0415
    import torch            # noqa: PLC0415
    from PIL import Image   # noqa: PLC0415

    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]
    return tensor


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class GeminiImageInteractionsNode:
    """
    ComfyUI node that calls the Google Gemini Interactions API to generate or
    edit images.  Returns both the generated image tensor and the interaction
    ID, which a downstream node can pass as `previous_interaction_id` to chain
    multi-turn refinements.

    API reference: https://ai.google.dev/gemini-api/docs/image-generation
    Endpoint: POST https://generativelanguage.googleapis.com/v1beta/interactions
    """

    CATEGORY = "aw/gemini"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "interaction_id")

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
                        "default": "GEMINI_API_KEY",
                    },
                ),
                "model": (
                    [
                        "gemini-3-pro-image",
                        "gemini-3.1-flash-image",
                        "gemini-3.1-flash-lite-image",
                        "gemini-2.5-flash-image",
                    ],
                    {
                        "default": "gemini-3-pro-image",
                    },
                ),
                "aspect_ratio": (
                    # All ratios documented for the Interactions API
                    ["16:9", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "21:9"],
                    {
                        "default": "16:9",
                    },
                ),
                "image_size": (
                    # Documented sizes; 512px is 3.1-flash-lite only, 2K/4K not for lite.
                    # The node accepts all options; the API will reject invalid combos.
                    ["2K", "1K", "4K", "512px"],
                    {
                        "default": "2K",
                    },
                ),
                "mime_type": (
                    ["image/png", "image/jpeg"],
                    {
                        "default": "image/png",
                    },
                ),
                # seed: the Interactions API does not document a seed parameter.
                # Accepted here for compatibility with the WorkflowParameterMapper
                # which injects seed into all nodes — the value is silently ignored.
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
                # Input image batch (tensor [N,H,W,3]).  Frames are serialised
                # to PNG and appended to the `input` array as type "image" blocks.
                "images": ("IMAGE",),
                # When non-empty, sent as `previous_interaction_id` to continue a
                # prior interaction (multi-turn refinement).
                "previous_interaction_id": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "system_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                # thinking_level is only forwarded for 3.1-family models (per doc).
                # For other models the field is accepted here but silently dropped.
                "thinking_level": (
                    ["minimal", "high"],
                    {
                        "default": "minimal",
                    },
                ),
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
        aspect_ratio: str,
        image_size: str,
        mime_type: str,
        seed: int,  # noqa: ARG002 — intentionally unused; see INPUT_TYPES comment
        images=None,
        previous_interaction_id: str = "",
        system_instruction: str = "",
        thinking_level: str = "minimal",
    ):
        # ---- resolve key ---------------------------------------------------
        resolved_key = _resolve_api_key(api_key)
        if not resolved_key:
            raise RuntimeError(
                "GeminiImageInteractionsNode: api_key resolved to an empty string. "
                "Set GEMINI_API_KEY in the ComfyUI .env or provide the key directly."
            )

        # ---- build input array ---------------------------------------------
        input_blocks = []

        # Optional system instruction — prepended as a text block before the prompt
        if system_instruction and system_instruction.strip():
            input_blocks.append({"type": "text", "text": system_instruction.strip()})

        # User prompt (required)
        input_blocks.append({"type": "text", "text": prompt})

        # Optional reference images
        if images is not None:
            import torch  # noqa: PLC0415 — always present in ComfyUI

            # images is a batch tensor [N,H,W,3]
            if images.ndim == 3:
                images = images.unsqueeze(0)  # treat single H×W×3 as batch of 1

            n_frames = images.shape[0]
            limit = _MODEL_IMAGE_LIMITS.get(model, _DEFAULT_IMAGE_LIMIT)

            if n_frames > limit:
                warnings.warn(
                    f"GeminiImageInteractionsNode: {n_frames} images supplied but "
                    f"model '{model}' supports at most {limit}. "
                    f"Truncating to {limit} frames.",
                    stacklevel=2,
                )
                images = images[:limit]
                n_frames = limit

            for i in range(n_frames):
                frame_tensor = images[i]  # [H,W,3]
                b64 = _tensor_to_png_b64(frame_tensor)
                input_blocks.append(
                    {
                        "type": "image",
                        "mime_type": "image/png",
                        "data": b64,
                    }
                )

        # ---- build request body --------------------------------------------
        body = {
            "model": model,
            "input": input_blocks,
            "response_format": {
                "type": "image",
                "mime_type": mime_type,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
            },
        }

        if previous_interaction_id and previous_interaction_id.strip():
            body["previous_interaction_id"] = previous_interaction_id.strip()

        # thinking_level: only send for models that support it (3.1-family)
        if model in _THINKING_LEVEL_MODELS:
            body["generation_config"] = {"thinking_level": thinking_level}

        # NOTE: seed is not forwarded — the Interactions API does not document
        # a seed parameter anywhere in generation_config or the request body.

        # ---- HTTP request --------------------------------------------------
        headers = {
            "x-goog-api-key": resolved_key,
            "Content-Type": "application/json",
        }

        response = requests.post(
            INTERACTIONS_ENDPOINT,
            headers=headers,
            json=body,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            snippet = response.text[:400]
            raise RuntimeError(
                f"GeminiImageInteractionsNode: API returned HTTP {response.status_code}. "
                f"Body (truncated): {snippet}"
            )

        # ---- parse response ------------------------------------------------
        data = response.json()

        # Extract interaction ID — documented as `id` on the response object.
        interaction_id = data.get("id", "")

        # Find the first image content block by iterating steps.
        # Response shape (per doc):
        #   { "id": "<str>", "steps": [ { "type": "model_output",
        #     "content": [ {"type":"image","data":"<b64>","mime_type":"..."}, ... ] } ] }
        image_b64 = None
        fallback_text_parts = []

        for step in data.get("steps", []):
            if step.get("type") != "model_output":
                continue
            for block in step.get("content", []):
                if block.get("type") == "image" and block.get("data"):
                    image_b64 = block["data"]
                    break
                if block.get("type") == "text":
                    fallback_text_parts.append(block.get("text", ""))
            if image_b64:
                break

        if not image_b64:
            # Also check top-level convenience property if present
            output_image = data.get("output_image", {})
            if isinstance(output_image, dict) and output_image.get("data"):
                image_b64 = output_image["data"]

        if not image_b64:
            text_preview = " ".join(fallback_text_parts)[:300]
            raise RuntimeError(
                f"GeminiImageInteractionsNode: API response contained no image data. "
                f"The model may have returned text only. "
                f"Text preview (first 300 chars): {text_preview!r}"
            )

        # ---- decode to ComfyUI tensor --------------------------------------
        image_tensor = _png_b64_to_tensor(image_b64)  # [1, H, W, 3] float32

        return (image_tensor, interaction_id)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GeminiImageInteractionsNode": GeminiImageInteractionsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageInteractionsNode": "Gemini Image (Interactions API)",
}
