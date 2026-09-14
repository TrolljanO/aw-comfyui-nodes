"""
AwKlingVideoNode — ComfyUI custom node for Kling AI video generation.

REST surface:
  POST https://api-singapore.klingai.com/v1/videos/text2video
  POST https://api-singapore.klingai.com/v1/videos/image2video
  GET  https://api-singapore.klingai.com/v1/videos/{route}/{task_id}

Auth: JWT HS256, generated fresh per request.
  header  {"alg":"HS256","typ":"JWT"}
  payload {"iss":<access_key_id>, "exp":now+1800, "nbf":now-5}
  signed with HMAC-SHA256 using secret_access_key.

Doc reference: Kling AI skill / https://klingai.com/docs
"""

import base64
import hashlib
import hmac
import io
import json
import os
import re
import time

# requests and ComfyUI libs (folder_paths, torch, numpy, PIL) are imported
# lazily inside functions — pack convention to allow module import in
# environments where they are absent (e.g. local compile checks).

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_API_BASE = "https://api-singapore.klingai.com"
_POLL_INTERVAL_SECONDS = 10  # seconds between status-poll requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _resolve_credential(raw_value: str, field: str, required: bool = True) -> str:
    """
    Resolve a credential input following the pack convention: the input carries
    either the NAME of an environment variable or the literal key material.

    Falha ALTO quando o valor tem forma de nome de env var e a env nao existe.
    O fallback silencioso anterior devolvia a propria string ("KLING_SECRET_KEY")
    como se fosse o segredo, assinava o JWT com ela, e o Kling respondia
    401 code=1002 — um erro que aponta para a conta, nao para a configuracao.
    Meia hora de investigacao na direcao errada; ver o relato em 11/09/2026.
    """
    stripped = raw_value.strip()

    if not stripped:
        if not required:
            return ""
        raise RuntimeError(
            f"AwKlingVideoNode: o input '{field}' esta vazio. Informe o NOME de uma "
            f"variavel de ambiente (ex.: KLING_ACCESS_KEY) ou cole a credencial."
        )

    if _ENV_NAME_RE.match(stripped):
        env_val = os.environ.get(stripped, "")
        if not env_val:
            if not required:
                return ""
            raise RuntimeError(
                f"AwKlingVideoNode: o input '{field}' vale '{stripped}', que tem forma de "
                f"nome de variavel de ambiente, mas essa variavel NAO esta definida no "
                f"processo do ComfyUI. O node nao vai assinar o token com esse texto. "
                f"Defina a env no container, ou ligue este input a um node de secret, "
                f"ou cole a credencial literal. Lembre que o Kling exige o PAR "
                f"access_key + secret_key — uma chave sozinha nao autentica."
            )
        return env_val

    return stripped


def _gen_jwt(access_key: str, secret_key: str) -> str:
    """
    Generate a fresh HS256 JWT for the Kling API.

    Per Kling auth contract:
      header  {"alg":"HS256","typ":"JWT"}
      payload {"iss":<access_key_id>, "exp":now+1800, "nbf":now-5}
      signed with HMAC-SHA256 using secret_access_key.
    """

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"iss": access_key, "exp": now + 1800, "nbf": now - 5}

    h = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h}.{p}"

    sig_bytes = hmac.new(
        secret_key.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    return f"{signing_input}.{_b64url(sig_bytes)}"


def _api_base() -> str:
    """Return the Kling API base URL, respecting the optional KLING_API_BASE env override."""
    return os.environ.get("KLING_API_BASE", _DEFAULT_API_BASE).rstrip("/")


def _tensor_to_png_b64(tensor) -> str:
    """
    Convert a ComfyUI IMAGE tensor (shape [H,W,3] or [1,H,W,3], float32 0–1)
    to a raw base64 PNG string — NO data-uri prefix, as required by Kling.
    """
    import numpy as np       # noqa: PLC0415 — always present in ComfyUI
    import torch             # noqa: PLC0415
    from PIL import Image    # noqa: PLC0415

    if tensor.ndim == 4:
        tensor = tensor[0]   # take first frame from batch
    arr = (tensor.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _kling_post(url: str, token: str, body: dict) -> dict:
    """
    POST to the Kling API with a ready Bearer token (JWT assinado, ou API key direta).
    Raises RuntimeError on:
      - non-200 HTTP status
      - response body where code != 0 (Kling returns errors via code even on HTTP 200)
    """
    import requests as req  # noqa: PLC0415

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = req.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code != 200:
        snippet = resp.text[:400]
        raise RuntimeError(
            f"AwKlingVideoNode: POST {url} returned HTTP {resp.status_code}. "
            f"Body (truncated): {snippet}"
        )
    data = resp.json()
    # IMPORTANT: Kling signals API errors via code != 0 even on HTTP 200.
    # e.g. {"code":1000,"message":"signature is invalid"}
    if data.get("code", 0) != 0:
        raise RuntimeError(
            f"AwKlingVideoNode: API error — code={data.get('code')} "
            f"message={data.get('message', '(no message)')!r}"
        )
    return data


def _kling_get(url: str, token: str) -> dict:
    """
    GET from the Kling API with a ready Bearer token.
    Raises RuntimeError on HTTP error or API-level error (code != 0).
    """
    import requests as req  # noqa: PLC0415

    headers = {"Authorization": f"Bearer {token}"}
    resp = req.get(url, headers=headers, timeout=60)
    if resp.status_code != 200:
        snippet = resp.text[:400]
        raise RuntimeError(
            f"AwKlingVideoNode: GET {url} returned HTTP {resp.status_code}. "
            f"Body (truncated): {snippet}"
        )
    data = resp.json()
    if data.get("code", 0) != 0:
        raise RuntimeError(
            f"AwKlingVideoNode: API error — code={data.get('code')} "
            f"message={data.get('message', '(no message)')!r}"
        )
    return data


def _poll_until_done(
    route: str,
    task_id: str,
    access_key: str,
    secret_key: str,
    timeout_seconds: int,
) -> str:
    """
    Poll GET /v1/videos/{route}/{task_id} every _POLL_INTERVAL_SECONDS seconds
    until task_status is 'succeed' or 'failed'.

    Returns the video URL on success.
    Raises RuntimeError on failure or timeout.
    """
    poll_url = f"{_api_base()}/v1/videos/{route}/{task_id}"
    deadline = time.time() + timeout_seconds

    while True:
        if time.time() >= deadline:
            raise RuntimeError(
                f"AwKlingVideoNode: task {task_id!r} timed out after "
                f"{timeout_seconds}s. The video was not ready within the "
                f"configured limit. Increase timeout_seconds (currently "
                f"{timeout_seconds}) if the model needs more time."
            )

        resp_data = _kling_get(poll_url, token)
        task_data = resp_data.get("data", {})
        status = task_data.get("task_status", "")

        if status == "succeed":
            try:
                video_url = task_data["task_result"]["videos"][0]["url"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"AwKlingVideoNode: task {task_id!r} reported 'succeed' but "
                    f"could not extract video URL from response. "
                    f"Raw task_result: {task_data.get('task_result')!r}"
                ) from exc
            return video_url

        if status == "failed":
            fail_reason = task_data.get("task_status_msg", "(no reason provided)")
            raise RuntimeError(
                f"AwKlingVideoNode: task {task_id!r} failed. "
                f"Reason: {fail_reason!r}"
            )

        # Still running (e.g. "submitted", "processing") — wait and retry.
        time.sleep(_POLL_INTERVAL_SECONDS)


def _download_mp4(video_url: str) -> bytes:
    """Download the mp4 from the given URL and return raw bytes."""
    import requests as req  # noqa: PLC0415

    resp = req.get(video_url, timeout=180, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(
            f"AwKlingVideoNode: failed to download video from {video_url!r}. "
            f"HTTP {resp.status_code}."
        )
    return resp.content


def _save_mp4(mp4_bytes: bytes, filename_prefix: str) -> tuple:
    """
    Save mp4_bytes to ComfyUI's output directory.

    Uses a monotonic counter and O_CREAT|O_EXCL to atomically claim a
    unique filename.  Two concurrent replicas sharing the same EFS volume
    can never receive the same filename: if FileExistsError is raised,
    the counter is incremented and the next candidate is tried.

    NOTE: The prefix is expected to already be unique per job (injected by
    the WorkflowParameterMapper) to prevent EFS filename-collision races
    between concurrent replicas.  This atomic O_EXCL loop is the safety
    net — not a replacement for that good practice.

    Returns (filename, subfolder).
    """
    import folder_paths  # noqa: PLC0415 — always present in ComfyUI

    output_dir = folder_paths.get_output_directory()
    counter = 1
    while True:
        filename = f"{filename_prefix}_{counter:05d}.mp4"
        filepath = os.path.join(output_dir, filename)
        try:
            fd = os.open(filepath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            counter += 1
            continue
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(mp4_bytes)
        except Exception:
            # Writing failed — remove the empty placeholder so the slot is
            # not permanently reserved.
            try:
                os.unlink(filepath)
            except OSError:
                pass
            raise
        return filename, ""  # subfolder empty = root of output_dir


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------


class AwKlingVideoNode:
    """
    ComfyUI node that generates video using the Kling AI API.

    Mode selection:
      - No image connected → text-to-video  (POST /v1/videos/text2video)
      - image connected    → image-to-video (POST /v1/videos/image2video)

    The task is submitted asynchronously; this node polls at 10-second
    intervals until the video is ready, then downloads the mp4, writes it
    to ComfyUI's output directory, and returns the ui dict the finalizer
    expects.

    Credentials: KLING_ACCESS_KEY (access key ID) and KLING_SECRET_KEY
    (secret access key) — inputs accept the env-var name, following the
    pack convention.
    """

    CATEGORY = "aw/kling"
    FUNCTION = "generate"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A photorealistic architectural walkthrough",
                    },
                ),
                # Token Bearer PRONTO. Tem prioridade sobre access_key/secret_key, espelhando
                # a prioridade da skill oficial (KLING_TOKEN > AK/SK -> JWT). Serve para quem
                # so tem UMA credencial: cola aqui e o node nao assina JWT nenhum.
                "api_key": (
                    "STRING",
                    {
                        "default": "KLING_API_KEY",
                        "tooltip": (
                            "Token Bearer pronto (ou o NOME de uma env var que o contenha). "
                            "Se preenchido, access_key/secret_key sao ignorados."
                        ),
                    },
                ),
                "access_key": (
                    "STRING",
                    {
                        "default": "KLING_ACCESS_KEY",
                    },
                ),
                "secret_key": (
                    "STRING",
                    {
                        "default": "KLING_SECRET_KEY",
                    },
                ),
                # Free-form string — not a COMBO — so new model names don't
                # require a code change.  Known models as of 2026-09-11:
                # kling-v3 (default), kling-v2-6, kling-v3-omni, kling-video-o1
                "model_name": (
                    "STRING",
                    {
                        "default": "kling-v3",
                    },
                ),
                # text2video only; silently ignored in image2video mode.
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                # Kling API expects duration as a string (e.g. "5"), range 3–15.
                "duration": (
                    "STRING",
                    {
                        "default": "5",
                    },
                ),
                "mode": (
                    ["std", "pro"],
                    {
                        "default": "std",
                    },
                ),
                # aspect_ratio applies to text2video only; image2video inherits
                # the source image's aspect ratio.
                "aspect_ratio": (
                    ["16:9", "9:16", "1:1"],
                    {
                        "default": "16:9",
                    },
                ),
                "sound": (
                    ["off", "on"],
                    {
                        "default": "off",
                    },
                ),
                # CRITICAL: must be unique per job to prevent EFS filename-collision
                # races when multiple ComfyUI replicas share the same output volume.
                # The WorkflowParameterMapper injects a unique prefix per render job.
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "aw_kling",
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 600,
                        "min": 60,
                        "max": 3600,
                    },
                ),
                # seed: the Kling API does not use a seed parameter.
                # Accepted here for compatibility with the WorkflowParameterMapper,
                # which injects seed into all nodes.  Value is silently ignored.
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
                # If connected, the node switches to image-to-video mode.
                # Tensor shape [H,W,3] or [1,H,W,3]; first frame is used.
                # Converted to PNG base64 (no data-uri prefix) before sending.
                "image": ("IMAGE",),
                # Optional last/tail frame for image-to-video (constrains the
                # final frame of the generated clip).  Same tensor convention.
                "image_tail": ("IMAGE",),
            },
        }

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        api_key: str,
        access_key: str,
        secret_key: str,
        model_name: str,
        negative_prompt: str,
        duration: str,
        mode: str,
        aspect_ratio: str,
        sound: str,
        filename_prefix: str,
        timeout_seconds: int,
        seed: int,  # noqa: ARG002 — intentionally unused; see INPUT_TYPES comment
        image=None,
        image_tail=None,
    ):
        # ---- resolve credentials -------------------------------------------
        # Prioridade de credencial, igual a da skill oficial do Kling:
        #   1) api_key  -> usado DIRETO como Bearer, sem assinar nada
        #   2) AK + SK  -> JWT HS256 por request
        # O Bearer do header e o par AK/SK nao sao caminhos rivais: o JWT assinado com
        # AK/SK E o que vai no Bearer. Quem so tem uma credencial usa o caminho 1.
        raw_api_key = (api_key or "").strip()
        token = ""

        if raw_api_key:
            resolved = _resolve_credential(raw_api_key, "api_key", required=False)
            if resolved:
                token = resolved

        if not token:
            resolved_access = _resolve_credential(access_key, "access_key")
            resolved_secret = _resolve_credential(secret_key, "secret_key")
            token = _gen_jwt(resolved_access, resolved_secret)

        if not token:
            raise RuntimeError(
                "AwKlingVideoNode: nenhuma credencial utilizavel. Preencha api_key com um "
                "token Bearer pronto, OU access_key + secret_key para o node assinar o JWT."
            )
        base = _api_base()

        # ---- build request body and choose endpoint -----------------------
        if image is not None:
            # image-to-video mode
            route = "image2video"
            endpoint = f"{base}/v1/videos/image2video"

            body = {
                "model_name": model_name,
                "image": _tensor_to_png_b64(image),
                "prompt": prompt,
                "duration": duration,
                "mode": mode,
                "sound": sound,
                # aspect_ratio and negative_prompt are text2video-only params.
            }

            if image_tail is not None:
                body["image_tail"] = _tensor_to_png_b64(image_tail)

        else:
            # text-to-video mode
            route = "text2video"
            endpoint = f"{base}/v1/videos/text2video"

            body = {
                "model_name": model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "duration": duration,
                "mode": mode,
                "aspect_ratio": aspect_ratio,
                "sound": sound,
            }

        # ---- submit task ---------------------------------------------------
        submit_resp = _kling_post(endpoint, token, body)
        try:
            task_id = submit_resp["data"]["task_id"]
        except (KeyError, TypeError) as exc:
            body_snippet = str(submit_resp)[:300]
            raise RuntimeError(
                f"AwKlingVideoNode: API returned code=0 but task_id is missing. "
                f"Response body (truncated): {body_snippet}"
            ) from exc

        # ---- poll until video is ready ------------------------------------
        video_url = _poll_until_done(route, task_id, token, timeout_seconds)

        # ---- download mp4 -------------------------------------------------
        mp4_bytes = _download_mp4(video_url)

        # ---- write to ComfyUI output directory ----------------------------
        filename, subfolder = _save_mp4(mp4_bytes, filename_prefix)

        # ---- return ui dict (consumed by the finalizer / gallery) ---------
        # RenderJobFinalizer.php looks for outputs[nodeId]["video"] and treats
        # .mp4 files as video/mp4.  Shape must match exactly.
        return {"ui": {"video": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AwKlingVideoNode": AwKlingVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AwKlingVideoNode": "Kling Video (text2video / image2video)",
}
