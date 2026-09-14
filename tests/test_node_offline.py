"""
Offline test for GeminiImageInteractionsNode.

Monkeypatches requests.post with a canned doc-shaped response.
Stubs torch, numpy, and PIL when not available on the host.

Tests:
  1. image tensor shape/dtype/range (when torch available; mocked at boundary otherwise)
  2. interaction_id extraction from response['id']
  3. no-image response -> RuntimeError with text preview
  4. HTTP error -> RuntimeError with status code
  5. api_key env-var resolution
  6. thinking_level only forwarded for 3.1-family models
  7. previous_interaction_id forwarded in request body
  8. response_format always type "image"

Run with:  python3 test_node_offline.py
"""

import base64
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Detect available libraries and install stubs as needed BEFORE module import
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
_NUMPY_AVAILABLE = False
_PIL_AVAILABLE = False

try:
    import numpy as _np_real
    _NUMPY_AVAILABLE = True
except ImportError:
    pass

try:
    import torch as _torch_real
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image as _pil_real
    _PIL_AVAILABLE = True
except ImportError:
    pass

# ---- numpy stub ------------------------------------------------------------
if not _NUMPY_AVAILABLE:
    _np_stub = types.ModuleType("numpy")

    class _FakeArray:
        """Minimal ndarray stand-in."""
        def __init__(self, shape=(1, 64, 64, 3), dtype=None):
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = dtype or "uint8"

        def astype(self, dtype):
            a = _FakeArray(self.shape, dtype)
            return a

        def __truediv__(self, other):
            return _FakeArray(self.shape, "float32")

        def __mul__(self, other):
            return _FakeArray(self.shape, "uint8")

    _np_stub.array = lambda *a, **kw: _FakeArray()
    _np_stub.float32 = "float32"
    _np_stub.uint8 = "uint8"
    _np_stub.ndarray = _FakeArray
    sys.modules["numpy"] = _np_stub
    import numpy as _np_real  # re-import from stub

# ---- torch stub ------------------------------------------------------------
if not _TORCH_AVAILABLE:
    _torch_stub = types.ModuleType("torch")

    class _FakeTensor:
        """Minimal tensor stand-in."""
        def __init__(self, shape=(1, 64, 64, 3)):
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = _torch_stub.float32

        def clamp(self, *a, **kw):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return _np_real.array()

        def unsqueeze(self, dim):
            new_shape = list(self.shape)
            new_shape.insert(dim, 1)
            return _FakeTensor(tuple(new_shape))

        def __getitem__(self, idx):
            return _FakeTensor(self.shape[1:] if self.ndim > 1 else self.shape)

        def min(self):
            return 0.0

        def max(self):
            return 1.0

    class _FakeDtype:
        def __repr__(self):
            return "torch.float32"

    _torch_stub.float32 = _FakeDtype()
    _torch_stub.from_numpy = lambda arr: _FakeTensor((1, 64, 64, 3))
    _torch_stub.Tensor = _FakeTensor
    sys.modules["torch"] = _torch_stub

# ---- PIL stub --------------------------------------------------------------
if not _PIL_AVAILABLE:
    _pil_mod = types.ModuleType("PIL")
    _pil_image_mod = types.ModuleType("PIL.Image")

    class _FakeImage:
        def __init__(self, size=(64, 64), mode="RGB"):
            self.size = size
            self.mode = mode

        def convert(self, mode):
            return _FakeImage(self.size, mode)

        def save(self, buf, format=None):
            # Write a minimal valid 1×1 PNG into the buffer
            _1x1_png = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
                b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            buf.write(_1x1_png)

    _pil_image_mod.new = lambda mode, size, color=None: _FakeImage(size, mode)
    _pil_image_mod.fromarray = lambda arr, mode="RGB": _FakeImage(mode=mode)
    _pil_image_mod.open = lambda buf: _FakeImage()
    _pil_mod.Image = _pil_image_mod
    sys.modules["PIL"] = _pil_mod
    sys.modules["PIL.Image"] = _pil_image_mod

# ---------------------------------------------------------------------------
# Build a small valid PNG image as base64 for the canned response
# ---------------------------------------------------------------------------

def _make_png_b64() -> str:
    if _PIL_AVAILABLE:
        from PIL import Image  # noqa: PLC0415
        buf = io.BytesIO()
        img = Image.new("RGB", (64, 64), color=(128, 64, 200))
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        # Minimal valid 1×1 PNG
        _1x1_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return base64.b64encode(_1x1_png).decode("utf-8")


FAKE_IMAGE_B64 = _make_png_b64()
FAKE_INTERACTION_ID = "interactions/abc123xyz"

# Canned successful API response matching the documented structure:
# { "id": "<str>", "steps": [ { "type": "model_output",
#   "content": [ {"type":"image","data":"<b64>","mime_type":"..."} ] } ] }
CANNED_SUCCESS_RESPONSE = {
    "id": FAKE_INTERACTION_ID,
    "steps": [
        {
            "type": "model_output",
            "content": [
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": FAKE_IMAGE_B64,
                }
            ],
        }
    ],
}

# Canned text-only response (no image in any step)
CANNED_TEXT_ONLY_RESPONSE = {
    "id": "interactions/text_only",
    "steps": [
        {
            "type": "model_output",
            "content": [
                {
                    "type": "text",
                    "text": "I cannot generate an image for this request.",
                }
            ],
        }
    ],
}


def _make_fake_response(body: dict, status_code: int = 200):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


# ---------------------------------------------------------------------------
# Import the module under test (after stubs are in place)
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_THIS_DIR, "..", "nodes", "gemini_image_interactions.py")
_spec = importlib.util.spec_from_file_location("gemini_image_interactions", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GeminiImageInteractionsNode = _mod.GeminiImageInteractionsNode
_resolve_api_key = _mod._resolve_api_key


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGeminiImageInteractionsNode(unittest.TestCase):

    def _node(self):
        return GeminiImageInteractionsNode()

    # ------------------------------------------------------------------
    # 1. Successful image generation: tensor returned + interaction_id
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_successful_image_returns_tensor_and_interaction_id(self, mock_post):
        mock_post.return_value = _make_fake_response(CANNED_SUCCESS_RESPONSE, 200)

        node = self._node()
        result = node.generate(
            prompt="A beautiful building",
            api_key="FAKE_KEY_LITERAL",  # literal (not an env var name) → used as-is
            model="gemini-3.1-flash-image",
            aspect_ratio="16:9",
            image_size="2K",
            mime_type="image/jpeg",
            seed=42,  # silently ignored; API does not support seed
        )

        image_tensor, interaction_id = result

        # interaction_id must equal response["id"]
        self.assertEqual(
            interaction_id,
            FAKE_INTERACTION_ID,
            "interaction_id must equal response['id']",
        )

        # Tensor must not be None
        self.assertIsNotNone(image_tensor, "image_tensor must not be None")

        if _TORCH_AVAILABLE and _NUMPY_AVAILABLE and _PIL_AVAILABLE:
            import torch  # noqa: PLC0415
            # Shape [1, H, W, 3]
            self.assertEqual(image_tensor.ndim, 4,
                             f"Expected 4D [1,H,W,3], got ndim={image_tensor.ndim}")
            self.assertEqual(image_tensor.shape[0], 1, "Batch dim must be 1")
            self.assertEqual(image_tensor.shape[3], 3, "Channel dim must be 3 (RGB)")
            self.assertEqual(image_tensor.dtype, torch.float32, "dtype must be float32")
            self.assertGreaterEqual(float(image_tensor.min()), 0.0)
            self.assertLessEqual(float(image_tensor.max()), 1.0)
        else:
            print("  (tensor shape/range check skipped — torch/numpy/PIL not available locally)")

    # ------------------------------------------------------------------
    # 2. No-image response → RuntimeError with text preview
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_text_only_response_raises_runtime_error(self, mock_post):
        mock_post.return_value = _make_fake_response(CANNED_TEXT_ONLY_RESPONSE, 200)

        node = self._node()
        with self.assertRaises(RuntimeError) as ctx:
            node.generate(
                prompt="An impossible prompt",
                api_key="FAKE_KEY",
                model="gemini-3-pro-image",
                aspect_ratio="1:1",
                image_size="1K",
                mime_type="image/jpeg",
                seed=0,
            )

        err_msg = str(ctx.exception)
        self.assertIn("no image data", err_msg.lower(),
                      "Error must mention missing image")
        self.assertIn("cannot generate", err_msg.lower(),
                      "Error must include model text preview")

    # ------------------------------------------------------------------
    # 3. HTTP error → RuntimeError with status code in message
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_http_error_raises_runtime_error(self, mock_post):
        mock_post.return_value = _make_fake_response(
            {"error": {"message": "API key not valid"}}, status_code=400
        )

        node = self._node()
        with self.assertRaises(RuntimeError) as ctx:
            node.generate(
                prompt="Test",
                api_key="BAD_KEY",
                model="gemini-3.1-flash-image",
                aspect_ratio="16:9",
                image_size="1K",
                mime_type="image/jpeg",
                seed=0,
            )

        self.assertIn("400", str(ctx.exception),
                      "Error must include HTTP 400 status code")

    # ------------------------------------------------------------------
    # 4. api_key env-var resolution
    # ------------------------------------------------------------------

    def test_api_key_resolved_from_env(self):
        os.environ["TEST_GEMINI_OFFLINE"] = "real_secret_value"
        try:
            resolved = _resolve_api_key("TEST_GEMINI_OFFLINE")
            self.assertEqual(resolved, "real_secret_value")
        finally:
            del os.environ["TEST_GEMINI_OFFLINE"]

    def test_api_key_literal_when_not_env_var(self):
        # Mixed case / dashes → treated as literal, not an env var name
        resolved = _resolve_api_key("my-actual-api-key-literal")
        self.assertEqual(resolved, "my-actual-api-key-literal")

    # ------------------------------------------------------------------
    # 5. thinking_level sent only for 3.1-family models
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_thinking_level_sent_only_for_flash31(self, mock_post):
        mock_post.return_value = _make_fake_response(CANNED_SUCCESS_RESPONSE, 200)

        node = self._node()
        # 3.1 Flash → generation_config must be present with thinking_level
        node.generate(
            prompt="Test",
            api_key="FAKE",
            model="gemini-3.1-flash-image",
            aspect_ratio="16:9",
            image_size="2K",
            mime_type="image/jpeg",
            seed=0,
            thinking_level="high",
        )
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        self.assertIn("generation_config", body)
        self.assertEqual(body["generation_config"]["thinking_level"], "high")

        # 3 Pro → generation_config must NOT be present
        mock_post.reset_mock()
        mock_post.return_value = _make_fake_response(CANNED_SUCCESS_RESPONSE, 200)
        node.generate(
            prompt="Test",
            api_key="FAKE",
            model="gemini-3-pro-image",
            aspect_ratio="16:9",
            image_size="2K",
            mime_type="image/jpeg",
            seed=0,
            thinking_level="high",
        )
        _, kwargs2 = mock_post.call_args
        self.assertNotIn(
            "generation_config", kwargs2.get("json", {}),
            "generation_config must NOT be sent for gemini-3-pro-image",
        )

    # ------------------------------------------------------------------
    # 6. previous_interaction_id forwarded in request body
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_previous_interaction_id_forwarded(self, mock_post):
        mock_post.return_value = _make_fake_response(CANNED_SUCCESS_RESPONSE, 200)

        node = self._node()
        node.generate(
            prompt="Refine the image",
            api_key="FAKE",
            model="gemini-3.1-flash-image",
            aspect_ratio="16:9",
            image_size="2K",
            mime_type="image/jpeg",
            seed=0,
            previous_interaction_id="interactions/prev456",
        )
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        self.assertEqual(body.get("previous_interaction_id"), "interactions/prev456")

    # ------------------------------------------------------------------
    # 7. response_format always type "image"
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_response_format_always_image_type(self, mock_post):
        mock_post.return_value = _make_fake_response(CANNED_SUCCESS_RESPONSE, 200)

        node = self._node()
        node.generate(
            prompt="Generate image",
            api_key="FAKE",
            model="gemini-3-pro-image",
            aspect_ratio="1:1",
            image_size="1K",
            mime_type="image/jpeg",
            seed=0,
        )
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        self.assertEqual(body["response_format"]["type"], "image")
        self.assertEqual(body["response_format"]["mime_type"], "image/jpeg")
        self.assertEqual(body["response_format"]["aspect_ratio"], "1:1")
        self.assertEqual(body["response_format"]["image_size"], "1K")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"torch available: {_TORCH_AVAILABLE}")
    print(f"numpy available: {_NUMPY_AVAILABLE}")
    print(f"PIL  available:  {_PIL_AVAILABLE}")
    print()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGeminiImageInteractionsNode)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


class TestOutputMimeTypeGuard(unittest.TestCase):
    """
    A Interactions API so aceita image/jpeg em response_format.mime_type. O default do
    node era image/png, entao todo grafo novo montado na UI nascia quebrado com HTTP
    400 — erro que chegava truncado ao usuario. Ver vault: "Interactions API so devolve JPEG".
    """

    def test_default_do_combo_e_jpeg(self):
        spec = _mod.GeminiImageInteractionsNode.INPUT_TYPES()["required"]["mime_type"]
        opcoes, cfg = spec[0], spec[1]
        self.assertEqual(cfg["default"], "image/jpeg")
        self.assertEqual(opcoes[0], "image/jpeg")

    def test_png_continua_selecionavel_para_nao_quebrar_grafo_salvo(self):
        opcoes = _mod.GeminiImageInteractionsNode.INPUT_TYPES()["required"]["mime_type"][0]
        self.assertIn("image/png", opcoes)
