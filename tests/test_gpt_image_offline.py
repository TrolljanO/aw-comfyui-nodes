"""
Offline tests for AwGptImageNode.

Coverage:
  1. model STRING passa verbatim para o payload (o ponto central do node)
  2. /edits escolhido quando image_1 conectado; /generations quando sem imagens
  3. Erro HTTP vira RuntimeError com o status code na mensagem
  4. Resolucao de api_key via env-var (convencao do pack)

Design de isolamento:
  - Nenhum stub e instalado globalmente em sys.modules para torch/numpy/PIL.
  - Testes que envolvem tensores usam patch.object nos helpers internos
    (_tensor_to_png_bytes, _b64_to_tensor) e patch.dict em sys.modules["torch"]
    apenas durante a execucao do metodo, sem poluir outros testes.

Execucao: python3 -m pytest tests/test_gpt_image_offline.py -v
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
# Canned PNG minimo para simular resposta da API
# ---------------------------------------------------------------------------

_1x1_png = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

_CANNED_PNG_B64 = base64.b64encode(_1x1_png).decode("utf-8")

CANNED_IMAGE_RESPONSE = {"data": [{"b64_json": _CANNED_PNG_B64}]}


def _make_fake_response(body, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


# ---------------------------------------------------------------------------
# Carrega o modulo sob teste
# aw_gpt_image.py importa `requests` no topo mas numpy/torch/PIL lazily —
# o modulo carrega sem nenhum stub de tensor.
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_THIS_DIR, "..", "nodes", "aw_gpt_image.py")
_spec = importlib.util.spec_from_file_location("aw_gpt_image", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

AwGptImageNode = _mod.AwGptImageNode
_resolve_api_key = _mod._resolve_api_key
_EDITS_ENDPOINT = _mod._EDITS_ENDPOINT
_GENERATIONS_ENDPOINT = _mod._GENERATIONS_ENDPOINT


# ---------------------------------------------------------------------------
# Fixture de torch minimal para testes que chegam no torch.cat
# ---------------------------------------------------------------------------

def _make_torch_stub():
    """Retorna um modulo torch minimal que sobrevive apenas ao bloco with."""
    stub = types.ModuleType("torch")
    stub.cat = lambda tensors, dim=0: tensors[0]
    return stub


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

class TestAwGptImageNode(unittest.TestCase):

    def _node(self):
        return AwGptImageNode()

    # ------------------------------------------------------------------
    # 1. model STRING passa verbatim para o payload /generations
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_model_string_passes_to_generations_payload(self, mock_post):
        """model deve aparecer verbatim no JSON body de /generations."""
        mock_post.return_value = _make_fake_response(CANNED_IMAGE_RESPONSE)
        node = self._node()

        # Sem imagem: vai para /generations (JSON body)
        with patch.dict(sys.modules, {"torch": _make_torch_stub()}), \
             patch.object(_mod, "_b64_to_tensor", return_value=MagicMock()):
            node.generate(
                prompt="generate an image",
                api_key="FAKE_KEY_LITERAL",
                model="gpt-image-2.5-flare",  # modelo futuro, fora de qualquer COMBO
                background="auto",
                size="1024x1024",
                quality="auto",
                output_format="png",
                output_compression=100,
                n_images=1,
                seed=0,
            )

        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        self.assertEqual(
            body["model"], "gpt-image-2.5-flare",
            "model deve passar verbatim para o payload (STRING, nao COMBO)",
        )

    @patch("requests.post")
    def test_model_string_passes_to_edits_payload(self, mock_post):
        """model deve aparecer verbatim no form data de /edits."""
        mock_post.return_value = _make_fake_response(CANNED_IMAGE_RESPONSE)
        node = self._node()
        fake_image = MagicMock()

        with patch.dict(sys.modules, {"torch": _make_torch_stub()}), \
             patch.object(_mod, "_tensor_to_png_bytes", return_value=_1x1_png), \
             patch.object(_mod, "_b64_to_tensor", return_value=MagicMock()):
            node.generate(
                prompt="edit this image",
                api_key="FAKE_KEY_LITERAL",
                model="gpt-image-2.5-sunburst",
                background="auto",
                size="1024x1024",
                quality="high",
                output_format="png",
                output_compression=100,
                n_images=1,
                seed=0,
                image_1=fake_image,
            )

        _, kwargs = mock_post.call_args
        form_data = kwargs.get("data", {})
        self.assertEqual(
            form_data["model"], "gpt-image-2.5-sunburst",
            "model deve passar verbatim no form data de /edits",
        )

    # ------------------------------------------------------------------
    # 2. /generations quando sem imagens; /edits quando ha imagem
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_no_image_calls_generations_endpoint(self, mock_post):
        """/generations escolhido quando nenhum image_N esta conectado."""
        mock_post.return_value = _make_fake_response(CANNED_IMAGE_RESPONSE)
        node = self._node()

        with patch.dict(sys.modules, {"torch": _make_torch_stub()}), \
             patch.object(_mod, "_b64_to_tensor", return_value=MagicMock()):
            node.generate(
                prompt="generate",
                api_key="FAKE_KEY",
                model="gpt-image-2",
                background="auto",
                size="1024x1024",
                quality="auto",
                output_format="png",
                output_compression=100,
                n_images=1,
                seed=0,
            )

        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, _GENERATIONS_ENDPOINT,
                         "sem imagens deve chamar /generations")
        _, kwargs = mock_post.call_args
        self.assertIn("json", kwargs, "/generations deve usar json= body")
        self.assertNotIn("files", kwargs, "/generations nao deve ter files=")

    @patch("requests.post")
    def test_with_image_calls_edits_endpoint(self, mock_post):
        """/edits escolhido quando image_1 esta conectado."""
        mock_post.return_value = _make_fake_response(CANNED_IMAGE_RESPONSE)
        node = self._node()
        fake_image = MagicMock()

        with patch.dict(sys.modules, {"torch": _make_torch_stub()}), \
             patch.object(_mod, "_tensor_to_png_bytes", return_value=_1x1_png), \
             patch.object(_mod, "_b64_to_tensor", return_value=MagicMock()):
            node.generate(
                prompt="edit this",
                api_key="FAKE_KEY",
                model="gpt-image-2",
                background="auto",
                size="1024x1024",
                quality="auto",
                output_format="png",
                output_compression=100,
                n_images=1,
                seed=0,
                image_1=fake_image,
            )

        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, _EDITS_ENDPOINT,
                         "com image_1 deve chamar /edits")
        _, kwargs = mock_post.call_args
        self.assertIn("files", kwargs, "/edits deve ter files= para multipart")

    # ------------------------------------------------------------------
    # 3. Erro HTTP vira RuntimeError com status code
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_http_error_raises_runtime_error(self, mock_post):
        mock_post.return_value = _make_fake_response(
            {"error": {"message": "Invalid API key."}}, status_code=401
        )
        node = self._node()
        with self.assertRaises(RuntimeError) as ctx:
            node.generate(
                prompt="test",
                api_key="BAD_KEY",
                model="gpt-image-2",
                background="auto",
                size="1024x1024",
                quality="auto",
                output_format="png",
                output_compression=100,
                n_images=1,
                seed=0,
            )
        self.assertIn("401", str(ctx.exception),
                      "RuntimeError deve mencionar o status HTTP 401")

    # ------------------------------------------------------------------
    # 4. Resolucao de api_key via env-var
    # ------------------------------------------------------------------

    def test_api_key_resolved_from_env(self):
        os.environ["TEST_GPT_IMAGE_OFFLINE_KEY"] = "sk-real-secret-from-env"
        try:
            resolved = _resolve_api_key("TEST_GPT_IMAGE_OFFLINE_KEY")
            self.assertEqual(resolved, "sk-real-secret-from-env")
        finally:
            del os.environ["TEST_GPT_IMAGE_OFFLINE_KEY"]

    def test_api_key_literal_passes_through(self):
        """Valor com minusculas/hifens e tratado como literal, nao env-var."""
        resolved = _resolve_api_key("sk-some-literal-key")
        self.assertEqual(resolved, "sk-some-literal-key")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAwGptImageNode)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
