"""
Offline tests for AwKlingVideoNode.

Coverage:
  1. _gen_jwt: monta JWT com iss/exp/nbf corretos e assinatura HS256 valida
  2. _kling_post: code!=0 com HTTP 200 levanta RuntimeError (nao retorna silenciosamente)
  3. _kling_post: HTTP 4xx levanta RuntimeError
  4. generate(): devolve {"ui": {"video": [{filename, subfolder, type}]}}
  5. generate(): access_key vazio levanta RuntimeError antes de qualquer chamada HTTP
  6. generate(): secret_key vazio levanta RuntimeError antes de qualquer chamada HTTP

Design de isolamento:
  - aw_kling_video.py so tem imports stdlib no topo — carrega sem stubs.
  - folder_paths e adicionado a sys.modules apenas se ausente (nao sobrescreve).
  - numpy/torch/PIL nao sao instalados globalmente; o teste de generate() usa
    texto-para-video (sem imagem), evitando o pipeline de tensores.

Execucao: python3 -m pytest tests/test_kling_video_offline.py -v
"""

import base64
import hashlib
import hmac
import importlib.util
import json
import os
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub folder_paths se ainda nao estiver em sys.modules
# (importado lazily dentro de _save_mp4; stub preventivo)
# ---------------------------------------------------------------------------
if "folder_paths" not in sys.modules:
    _folder_paths_stub = types.ModuleType("folder_paths")
    _folder_paths_stub.get_output_directory = lambda: "/tmp/fake_kling_output"
    sys.modules["folder_paths"] = _folder_paths_stub

# ---------------------------------------------------------------------------
# Carrega o modulo sob teste
# aw_kling_video.py tem apenas imports stdlib no topo — carrega sem stubs.
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_THIS_DIR, "..", "nodes", "aw_kling_video.py")
_spec = importlib.util.spec_from_file_location("aw_kling_video", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

AwKlingVideoNode = _mod.AwKlingVideoNode
_gen_jwt = _mod._gen_jwt
_kling_post = _mod._kling_post


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_jwt(token: str):
    """
    Decodifica um JWT sem verificar a assinatura.
    Retorna (header_dict, payload_dict, signature_bytes, signing_input_str).
    """
    parts = token.split(".")
    assert len(parts) == 3, f"JWT deve ter 3 partes, teve {len(parts)}"

    def _b64url_decode(s: str) -> bytes:
        pad = (4 - len(s) % 4) % 4
        return base64.urlsafe_b64decode(s + "=" * pad)

    header = json.loads(_b64url_decode(parts[0]))
    payload = json.loads(_b64url_decode(parts[1]))
    sig = _b64url_decode(parts[2])
    signing_input = f"{parts[0]}.{parts[1]}"
    return header, payload, sig, signing_input


# ---------------------------------------------------------------------------
# TestGenJwt
# ---------------------------------------------------------------------------

class TestGenJwt(unittest.TestCase):
    """Testa a funcao _gen_jwt() — usada em toda chamada a API Kling."""

    def test_jwt_claims_iss_exp_nbf(self):
        """iss == access_key; exp ~= now+1800; nbf ~= now-5."""
        before = int(time.time())
        token = _gen_jwt("test_access_key_123", "any_secret")
        after = int(time.time())

        _, payload, _, _ = _decode_jwt(token)

        self.assertEqual(payload["iss"], "test_access_key_123",
                         "iss deve ser o access_key")
        self.assertGreaterEqual(payload["exp"], before + 1800,
                                "exp deve ser pelo menos now+1800")
        self.assertLessEqual(payload["exp"], after + 1800 + 1,
                             "exp nao deve estar muito no futuro")
        self.assertLessEqual(payload["nbf"], before,
                             "nbf nao deve ser no futuro")
        self.assertGreaterEqual(payload["nbf"], before - 10,
                                "nbf nao deve estar muito no passado")

    def test_jwt_header_declares_hs256(self):
        """Header do JWT deve declarar alg=HS256 e typ=JWT."""
        token = _gen_jwt("acc", "sec")
        header, _, _, _ = _decode_jwt(token)
        self.assertEqual(header["alg"], "HS256", "alg deve ser HS256")
        self.assertEqual(header["typ"], "JWT", "typ deve ser JWT")

    def test_jwt_signature_is_valid_hs256(self):
        """Assinatura deve ser HMAC-SHA256('<header>.<payload>', secret)."""
        secret = "my_kling_secret_key"
        token = _gen_jwt("my_access_id", secret)
        _, _, sig_bytes, signing_input = _decode_jwt(token)

        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        self.assertEqual(sig_bytes, expected_sig,
                         "Assinatura deve ser HMAC-SHA256 sobre '<header>.<payload>'")


# ---------------------------------------------------------------------------
# TestKlingPost
# ---------------------------------------------------------------------------

class TestKlingPost(unittest.TestCase):
    """Testa _kling_post() — o caller do POST da API Kling."""

    @patch("requests.post")
    def test_code_nonzero_with_http_200_raises_runtime_error(self, mock_post):
        """HTTP 200 + code!=0 deve levantar RuntimeError (nao retornar silenciosamente)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 1000, "message": "signature is invalid"}
        mock_resp.text = '{"code": 1000, "message": "signature is invalid"}'
        mock_post.return_value = mock_resp

        with self.assertRaises(RuntimeError) as ctx:
            _kling_post(
                "https://api-singapore.klingai.com/v1/videos/text2video",
                "access_key",
                "secret_key",
                {"model_name": "kling-v3", "prompt": "test"},
            )

        err = str(ctx.exception)
        self.assertIn("1000", err,
                      "RuntimeError deve incluir o codigo de erro da API")
        self.assertIn("signature is invalid", err,
                      "RuntimeError deve incluir a mensagem da API")

    @patch("requests.post")
    def test_http_non_200_raises_runtime_error(self, mock_post):
        """HTTP 4xx deve levantar RuntimeError com o status code."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp

        with self.assertRaises(RuntimeError) as ctx:
            _kling_post(
                "https://api-singapore.klingai.com/v1/videos/text2video",
                "a", "b", {},
            )

        self.assertIn("401", str(ctx.exception))


# ---------------------------------------------------------------------------
# TestAwKlingVideoNodeGenerate
# ---------------------------------------------------------------------------

class TestAwKlingVideoNodeGenerate(unittest.TestCase):
    """Testa o metodo generate() do AwKlingVideoNode."""

    def test_generate_returns_ui_video_dict_shape(self):
        """generate() deve retornar {"ui": {"video": [{filename, subfolder, type}]}}."""
        node = AwKlingVideoNode()

        fake_submit_resp = {"data": {"task_id": "task_abc123"}}
        fake_video_url = "https://cdn.klingai.com/video/abc.mp4"
        fake_mp4_bytes = b"\x00\x00\x00\x1cftypisommp42\x00\x00\x00\x00"
        fake_filename = "aw_kling_00001.mp4"
        fake_subfolder = ""

        with patch.object(_mod, "_kling_post", return_value=fake_submit_resp), \
             patch.object(_mod, "_poll_until_done", return_value=fake_video_url), \
             patch.object(_mod, "_download_mp4", return_value=fake_mp4_bytes), \
             patch.object(_mod, "_save_mp4", return_value=(fake_filename, fake_subfolder)):

            result = node.generate(
                prompt="A photorealistic walkthrough",
                access_key="ak-literal-AbC123",   # nao tem forma de env var -> literal
                secret_key="sk-literal-XyZ789",
                model_name="kling-v3",
                negative_prompt="",
                duration="5",
                mode="std",
                aspect_ratio="16:9",
                sound="off",
                filename_prefix="aw_kling",
                timeout_seconds=600,
                seed=0,
            )

        self.assertIsInstance(result, dict, "generate() deve retornar um dict")
        self.assertIn("ui", result, "Resultado deve ter chave 'ui'")
        self.assertIn("video", result["ui"], "'ui' deve ter chave 'video'")

        videos = result["ui"]["video"]
        self.assertIsInstance(videos, list, "'video' deve ser uma lista")
        self.assertEqual(len(videos), 1, "'video' deve ter exatamente 1 item")

        video = videos[0]
        self.assertIn("filename", video)
        self.assertIn("subfolder", video)
        self.assertIn("type", video)
        self.assertEqual(video["filename"], fake_filename)
        self.assertEqual(video["subfolder"], fake_subfolder)
        self.assertEqual(video["type"], "output")

    def test_missing_access_key_raises_runtime_error(self):
        """access_key vazio deve levantar RuntimeError antes de qualquer chamada HTTP."""
        node = AwKlingVideoNode()
        with self.assertRaises(RuntimeError) as ctx:
            node.generate(
                prompt="test",
                access_key="",
                secret_key="some_secret",
                model_name="kling-v3",
                negative_prompt="",
                duration="5",
                mode="std",
                aspect_ratio="16:9",
                sound="off",
                filename_prefix="test",
                timeout_seconds=60,
                seed=0,
            )
        self.assertIn("access_key", str(ctx.exception).lower(),
                      "RuntimeError deve mencionar access_key")

    def test_missing_secret_key_raises_runtime_error(self):
        """secret_key vazio deve levantar RuntimeError antes de qualquer chamada HTTP."""
        node = AwKlingVideoNode()
        with self.assertRaises(RuntimeError) as ctx:
            node.generate(
                prompt="test",
                access_key="ak-literal-AbC123",
                secret_key="",
                model_name="kling-v3",
                negative_prompt="",
                duration="5",
                mode="std",
                aspect_ratio="16:9",
                sound="off",
                filename_prefix="test",
                timeout_seconds=60,
                seed=0,
            )
        self.assertIn("secret_key", str(ctx.exception).lower(),
                      "RuntimeError deve mencionar secret_key")

    def test_generate_code0_without_task_id_raises_runtime_error_with_body(self):
        """generate(): code=0 mas sem data.task_id deve levantar RuntimeError
        com o corpo da resposta na mensagem (DEFEITO 1)."""
        node = AwKlingVideoNode()
        # API retorna code=0 mas data nao tem task_id
        fake_submit_resp = {"code": 0, "data": {}}

        with patch.object(_mod, "_kling_post", return_value=fake_submit_resp):
            with self.assertRaises(RuntimeError) as ctx:
                node.generate(
                    prompt="test",
                    access_key="ak-literal-AbC123",
                    secret_key="sk-literal-XyZ789",
                    model_name="kling-v3",
                    negative_prompt="",
                    duration="5",
                    mode="std",
                    aspect_ratio="16:9",
                    sound="off",
                    filename_prefix="aw_kling",
                    timeout_seconds=60,
                    seed=0,
                )

        err = str(ctx.exception)
        self.assertIn("task_id", err.lower(),
                      "RuntimeError deve mencionar task_id")
        # Corpo da resposta (truncado) deve aparecer na mensagem para o operador
        self.assertIn("data", err,
                      "RuntimeError deve incluir o corpo da resposta API")


# ---------------------------------------------------------------------------
# TestSaveMp4Atomic
# ---------------------------------------------------------------------------

class TestSaveMp4Atomic(unittest.TestCase):
    """Testa que _save_mp4 escolhe o proximo indice atomicamente (DEFEITO 2)."""

    def test_save_mp4_skips_existing_file_and_uses_next_index(self):
        """_save_mp4 deve usar O_CREAT|O_EXCL: se _00001 ja existe,
        deve gravar em _00002 sem colisao."""
        import tempfile
        _save_mp4 = _mod._save_mp4

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simula outra replica que ja ganhou o nome _00001
            first = os.path.join(tmpdir, "aw_kling_00001.mp4")
            with open(first, "wb") as f:
                f.write(b"occupied_by_other_replica")

            with patch.object(
                sys.modules["folder_paths"],
                "get_output_directory",
                return_value=tmpdir,
            ):
                filename, subfolder = _save_mp4(b"\x00\x01\x02", "aw_kling")

        self.assertEqual(
            filename, "aw_kling_00002.mp4",
            "Deve escolher o proximo indice quando _00001 ja esta ocupado",
        )
        self.assertEqual(subfolder, "")

    def test_save_mp4_uses_o_excl_flag(self):
        """_save_mp4 deve chamar os.open com os.O_EXCL para garantir atomicidade."""
        import tempfile
        _save_mp4 = _mod._save_mp4

        calls = []
        real_os_open = os.open

        def capturing_open(path, flags, *args, **kwargs):
            calls.append(flags)
            return real_os_open(path, flags, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                sys.modules["folder_paths"],
                "get_output_directory",
                return_value=tmpdir,
            ), patch.object(_mod.os, "open", side_effect=capturing_open):
                _save_mp4(b"data", "aw_kling")

        self.assertTrue(len(calls) >= 1, "os.open deve ter sido chamado")
        flags_used = calls[0]
        self.assertTrue(
            flags_used & os.O_EXCL,
            f"os.open deve usar O_EXCL (flags={flags_used:#o})",
        )
        self.assertTrue(
            flags_used & os.O_CREAT,
            f"os.open deve usar O_CREAT (flags={flags_used:#o})",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suites = [
        loader.loadTestsFromTestCase(TestGenJwt),
        loader.loadTestsFromTestCase(TestKlingPost),
        loader.loadTestsFromTestCase(TestAwKlingVideoNodeGenerate),
    ]
    runner = unittest.TextTestRunner(verbosity=2)
    for suite in suites:
        result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


class TestCredentialFailLoud(unittest.TestCase):
    """
    O fallback silencioso assinava o JWT com a propria string "KLING_SECRET_KEY" e o
    Kling devolvia 401 code=1002 "access key not found" — um erro que aponta para a
    conta, nao para a env faltando. Aconteceu de verdade em 11/09/2026.
    """

    def test_env_name_shape_without_env_raises_naming_the_field_and_var(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                _mod._resolve_credential("KLING_SECRET_KEY", "secret_key")
        msg = str(ctx.exception)
        self.assertIn("secret_key", msg)
        self.assertIn("KLING_SECRET_KEY", msg)

    def test_env_name_shape_with_env_set_returns_env_value(self):
        with patch.dict(os.environ, {"KLING_SECRET_KEY": "sk-real-value"}, clear=True):
            self.assertEqual(
                _mod._resolve_credential("KLING_SECRET_KEY", "secret_key"), "sk-real-value"
            )

    def test_literal_credential_passes_through_untouched(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _mod._resolve_credential("AbC123-real-key", "access_key"), "AbC123-real-key"
            )

    def test_empty_input_raises(self):
        with self.assertRaises(RuntimeError):
            _mod._resolve_credential("   ", "access_key")
