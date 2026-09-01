"""
FastAPI wrapper exposing the audio feature-extraction pipeline as HTTP.
Level 1 + Level 2 feature extraction only -- no scoring, feedback, or LLM
inference (see backend/api/pipeline.py, which owns the actual orchestration;
this module owns only HTTP concerns: upload handling, request/response
schemas, error mapping, and temp-file cleanup).

Local run (replace <your-token> with a value of your choosing -- this is
just a local shared secret between you and your own server, not the value
stored in Modal's secret store):
    export CLIENT_TOKEN=<your-token>
    uvicorn backend.api.main:app --reload --port 8000
    curl -X POST http://localhost:8000/extract \\
        -H "X-Client-Token: <your-token>" -F "file=@path/to/audio.mp3"

Modal deployment: see modal_app.py at the repo root, which serves this same
`app` object via @modal.asgi_app() -- no separate app definition there.
CLIENT_TOKEN is injected there from a Modal Secret rather than set directly;
see verify_client_token()'s docstring below for the auth this protects.
"""

import hmac
import logging
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.api import warmup
from backend.api.pipeline import NoSpeechDetectedError, extract_features

logger = logging.getLogger(__name__)

# Shared-secret auth: the caller must send this same value in the
# X-Client-Token header. Locally, export CLIENT_TOKEN yourself before
# running uvicorn; on Modal it comes from a Secret (see modal_app.py) so the
# real value never lives in source control or this file.
CLIENT_TOKEN_ENV_VAR = "CLIENT_TOKEN"

app = FastAPI(
    title="Ziqra.ai Audio Feature Extraction API",
    description=(
        "Upload an audio file, get back Level 1 (timeline/ground-truth) and "
        "Level 2 (Fluency/Pronunciation/MTI/Intonation/Engagement) features "
        "as JSON. Extraction only -- no scoring, feedback, or LLM inference."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def _begin_warmup() -> None:
    """
    Starts warm-up the instant the container's server comes up, in a
    background thread — so a container booted by nothing more than a health
    ping is loading models while it waits, instead of at the moment someone
    finally speaks. Deliberately not awaited: the server must answer /health
    immediately and report `warm: false` until it is ready.
    """
    warmup.start_warmup()

# ffmpeg (silero_vad._convert_to_wav) can decode far more than this list --
# this is a cheap, fast rejection of obviously-wrong uploads (e.g. a .txt
# file) before spending time launching a subprocess and loading models.
# Anything in this list that ffmpeg still can't decode fails cleanly via the
# RuntimeError handler below, so this allowlist is a fast-path, not the only
# validation.
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".opus", ".ogg", ".flac", ".webm"}

# Generous but bounded -- this is a feature-extraction endpoint for testing
# audio files, not a general-purpose upload service.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


class HealthResponse(BaseModel):
    status: str = "ok"
    warm: bool = Field(False, description="True once models are loaded and the JIT paths are compiled.")
    warmup_seconds: Optional[float] = Field(None, description="How long warm-up took in this container.")


class ExtractResponse(BaseModel):
    session_id: str = Field(..., description="Generated per request; not persisted anywhere (see pipeline.py).")
    level1: Dict[str, Any] = Field(..., description="Timeline/ground-truth features -- dataset_builder.build_timeline's output, unchanged.")
    level2: Dict[str, Any] = Field(..., description="Fluency/Pronunciation/MTI/Intonation/Engagement -- dataset_builder.build_features's output, unchanged.")
    meta: Dict[str, Any] = Field(..., description="filename, size_bytes, duration_seconds, speech_seconds, processing_seconds.")


def _save_upload(file: UploadFile, dest_path: str) -> int:
    """Stream the upload to dest_path, enforcing MAX_UPLOAD_BYTES and
    rejecting an empty file. Returns the byte count.

    Uses file.file (the underlying sync SpooledTemporaryFile) rather than
    the async UploadFile.read()/write() API: the /extract endpoint below is
    a plain `def`, not `async def`, so FastAPI runs it in a threadpool
    (necessary because extract_features() is a long blocking call, and an
    async endpoint doing blocking work would stall the whole event loop) --
    and a sync function can't `await`. file.file is always available
    synchronously regardless of the endpoint's async-ness.
    """
    size = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                block = file.file.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                out.write(block)
    finally:
        file.file.close()

    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return size


def verify_client_token(x_client_token: Optional[str] = Header(None, alias="X-Client-Token")) -> None:
    """FastAPI dependency: rejects the request unless X-Client-Token matches
    the server's CLIENT_TOKEN env var. Applied to /extract only -- /health
    stays open so uptime/liveness monitoring doesn't need a credential.

    The header is declared Optional (default None) rather than required by
    FastAPI's own Header(...) validation on purpose: a required header would
    make FastAPI itself return 422 for a missing header, a different status
    than the 401 used below for a wrong one -- treating "missing" and
    "wrong" identically as 401 is both simpler for callers and standard
    practice (never reveal *why* auth failed).

    hmac.compare_digest (constant-time) rather than `==`, so a mistyped
    token can't be brute-forced faster by an attacker timing how many
    leading characters matched.
    """
    expected = os.environ.get(CLIENT_TOKEN_ENV_VAR)
    if not expected:
        # Fail closed: a server with no token configured rejects every
        # request rather than silently accepting all of them --
        # misconfiguration must never be equivalent to "no auth required".
        raise HTTPException(status_code=500, detail="Server auth is not configured.")
    if not x_client_token or not hmac.compare_digest(x_client_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing client token.")


@app.get("/health", response_model=HealthResponse)
def health(wait: float = 0.0) -> HealthResponse:
    """
    Liveness, and now readiness: `warm` says whether this container has
    already paid its one-time model-load and JIT cost (~28s, measured — see
    backend/api/warmup.py).

    Requires no auth (see verify_client_token's docstring for why) and never
    blocks by default, so it stays a cheap liveness probe. `?wait=<seconds>`
    blocks up to that long for warm-up to finish, which is what a caller
    wanting a guaranteed-warm container should use — the app pre-warms at the
    start of onboarding, and before this the ping only booted a container and
    left the whole cost for the user's first real answer.
    """
    warmup.start_warmup()
    if wait > 0:
        warmup.wait_until_warm(min(wait, 120.0))
    state = warmup.status()
    return HealthResponse(status="ok", warm=bool(state["warm"]), warmup_seconds=state["seconds"])


@app.post("/extract", response_model=ExtractResponse, dependencies=[Depends(verify_client_token)])
def extract(file: UploadFile = File(...)) -> ExtractResponse:
    """
    Run the full pipeline on one uploaded audio file and return its
    Level 1 + Level 2 features.

    Errors:
        400 -- unsupported extension, empty upload, no speech detected in
               the audio, or ffmpeg couldn't decode the file.
        401 -- missing or wrong X-Client-Token header.
        413 -- upload exceeds MAX_UPLOAD_BYTES.
        500 -- unexpected failure (logged server-side with a traceback, not
               echoed to the client), or the server has no CLIENT_TOKEN
               configured at all.
    """
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Everything this request touches on disk lives under one temp dir --
    # the upload itself and the per-VAD-segment chunk WAVs pipeline.py
    # writes into chunks_dir. Removed unconditionally on the way out
    # (success, a handled error, or a crash), so no run can leak files,
    # and concurrent requests never share a directory.
    with tempfile.TemporaryDirectory(prefix="ziqra_extract_") as tmp_dir:
        audio_path = os.path.join(tmp_dir, f"upload{ext}")
        chunks_dir = os.path.join(tmp_dir, "chunks")
        size_bytes = _save_upload(file, audio_path)

        try:
            result = extract_features(audio_path, chunks_dir)
        except NoSpeechDetectedError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            # silero_vad._convert_to_wav raises RuntimeError on ffmpeg decode
            # failure -- bad/corrupt/unsupported audio content, not a server fault.
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}") from e
        except HTTPException:
            raise
        except Exception:
            logger.exception("Unhandled error extracting features for %r", filename)
            raise HTTPException(status_code=500, detail="Internal error while processing audio.")

    result["meta"]["filename"] = filename
    result["meta"]["size_bytes"] = size_bytes
    return ExtractResponse(**result)
