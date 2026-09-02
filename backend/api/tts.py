"""
Indian-English text-to-speech, served as its own FastAPI app.

Why this is separate from backend/api/main.py rather than another route on
it: that app's container holds Whisper large-v3 and wav2vec2 on the GPU and
pays ~28s of one-time load before it can do anything. Adding a TTS model to
the same process would grow that cold start and the GPU memory footprint of
every transcription container, for a feature transcription never uses. Two
functions, two images, two scaledown clocks (see modal_app.py).

    curl -X POST $URL/tts -H "X-Client-Token: <token>" \\
        -H 'content-type: application/json' \\
        -d '{"text":"Tell me about yourself."}' --output q.mp3

The interviewer's voice is the product requirement here: an app for people
preparing for interviews in Indian English should not be read to in an
American accent. Device TTS covers that only when the handset happens to have
an en-IN voice installed, which is why this exists as the switchable
alternative (TTS_PROVIDER in the Worker's admin settings).

## Caching is what makes this affordable

Interview questions come from a fixed script, so the same few dozen strings
are synthesised over and over. Every result is written to a Modal Volume
keyed by a hash of (model, voice, text), so the GPU runs once per distinct
line ever — across containers and deploys, since the Volume outlives both.
A cache hit costs a file read.

## The model is gated

ai4bharat/indic-parler-tts is Apache-2.0 but gated on the Hub: it needs the
terms accepted once on the account and an HF token in the container. Nothing
here fails at import or build time because of that — the model is loaded
lazily on the first miss, and a load failure is reported as 503 so the caller
can fall back to device TTS rather than leaving a candidate in silence.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CLIENT_TOKEN_ENV_VAR = "CLIENT_TOKEN"

MODEL_ID = os.environ.get("TTS_MODEL_ID", "ai4bharat/indic-parler-tts")

# The speaker is chosen by describing them in words — Parler's whole
# interface. Kept as one env-overridable string so the voice can be retuned
# without a code change or a cache-invalidating model swap.
#
# "Anjali" is one of indic-parler-tts's recommended named Indian English
# speakers; the rest of the sentence is what the model card asks for: the
# recording conditions and the delivery, not the words.
VOICE = os.environ.get(
    "TTS_VOICE",
    "Anjali speaks Indian English in a clear, warm, professional tone, at a measured pace, "
    "as if interviewing someone. The recording is very close-sounding and completely noise-free.",
)

# Where synthesised audio lives. A Modal Volume in deployment; anywhere
# writable locally.
CACHE_DIR = Path(os.environ.get("TTS_CACHE_DIR", "/cache/tts"))

# Long enough for any interview question or coaching line, short enough that a
# runaway request cannot occupy the GPU generating minutes of audio.
MAX_CHARS = 600

_model = None
_tokenizer = None
_description_tokenizer = None
_load_lock = threading.Lock()
_load_error: Optional[str] = None
_volume = None

app = FastAPI(
    title="Ziqra.ai Indian-English TTS",
    description="Synthesises one short line of Indian English speech, cached by content hash.",
    version="1.0.0",
)


class SpeakRequest(BaseModel):
    text: str = Field(..., description="The line to speak. One or two sentences.")
    voice: Optional[str] = Field(
        None,
        description="Overrides the speaker description. Changes the cache key, so use it sparingly.",
    )


def verify_client_token(x_client_token: Optional[str] = Header(None, alias="X-Client-Token")) -> None:
    """Same shared-secret check as the extraction API — see
    backend/api/main.py's verify_client_token for why missing and wrong are
    both 401, and why a server with no token configured fails closed."""
    expected = os.environ.get(CLIENT_TOKEN_ENV_VAR)
    if not expected:
        raise HTTPException(status_code=500, detail="Server auth is not configured.")
    if not x_client_token or not hmac.compare_digest(x_client_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing client token.")


def cache_key(text: str, voice: str) -> str:
    """Model and voice are in the key, not just the text: changing either
    changes how the line sounds, and a stale hit would be worse than a miss."""
    digest = hashlib.sha256("\x00".join([MODEL_ID, voice, text]).encode("utf-8")).hexdigest()
    return digest


def _load() -> None:
    """Loads the model on first use, once per container.

    Deliberately not done in the image build or at import: the model is gated,
    so a build that had to download it would fail for anyone without the terms
    accepted and a token, and this whole service would be undeployable. The
    failure is remembered so a container without access does not retry a
    multi-GB download on every request.
    """
    global _model, _tokenizer, _description_tokenizer, _load_error
    if _model is not None or _load_error is not None:
        return
    with _load_lock:
        if _model is not None or _load_error is not None:
            return
        started = time.time()
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            # Parler reads the speaker description with the text encoder's own
            # tokenizer, which is a different one from the prompt tokenizer.
            description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
            _model, _tokenizer, _description_tokenizer = model, tokenizer, description_tokenizer
            logger.info("tts: loaded %s on %s in %.1fs", MODEL_ID, device, time.time() - started)
        except Exception as err:  # noqa: BLE001 - reported to the caller, not raised
            _load_error = f"{type(err).__name__}: {err}"
            logger.error("tts: could not load %s: %s", MODEL_ID, _load_error)


def _to_mp3(samples, sampling_rate: int) -> bytes:
    """Float32 waveform -> mono mp3.

    mp3, not the wav the model produces: a 5-second question is ~430KB of
    44.1kHz wav against ~40KB at 64kbps, and this is played on Indian mobile
    data. ffmpeg is in the image (see modal_app.py) and reads raw floats on
    stdin, so nothing is written to disk twice.
    """
    import numpy as np

    raw = np.asarray(samples, dtype=np.float32).tobytes()
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(sampling_rate), "-ac", "1", "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1",
        ],
        input=raw,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', 'replace')[:300]}")
    return result.stdout


def synthesise(text: str, voice: str) -> bytes:
    _load()
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"The Indian-English voice is unavailable: {_load_error}. "
                "ai4bharat/indic-parler-tts is a gated model — accept its terms on the Hub and give the "
                "container an HF_TOKEN. Callers should fall back to device speech."
            ),
        )
    import torch

    device = next(_model.parameters()).device
    description_ids = _description_tokenizer(voice, return_tensors="pt").to(device)
    prompt_ids = _tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        generation = _model.generate(
            input_ids=description_ids.input_ids,
            attention_mask=description_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
    audio = generation.cpu().numpy().squeeze()
    return _to_mp3(audio, _model.config.sampling_rate)


@app.get("/health")
def health() -> dict:
    """Open, like the extraction API's: whether the voice is usable at all is
    exactly what a caller deciding between this and device speech needs, and
    it reveals nothing a token would protect."""
    return {
        "status": "ok",
        "model": MODEL_ID,
        "loaded": _model is not None,
        "load_error": _load_error,
        "cached_lines": len(list(CACHE_DIR.glob("*.mp3"))) if CACHE_DIR.is_dir() else 0,
    }


@app.post("/tts", dependencies=[Depends(verify_client_token)])
def tts(request: SpeakRequest) -> Response:
    text = " ".join(request.text.split())
    if not text:
        raise HTTPException(status_code=400, detail="Nothing to say.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Text is longer than {MAX_CHARS} characters.")
    voice = " ".join((request.voice or VOICE).split())

    key = cache_key(text, voice)
    path = CACHE_DIR / f"{key}.mp3"
    if path.is_file():
        return Response(
            content=path.read_bytes(),
            media_type="audio/mpeg",
            headers={"X-Cache": "hit", "X-Cache-Key": key},
        )

    started = time.time()
    audio = synthesise(text, voice)
    took = time.time() - started

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Written under a temporary name and moved into place: a container
        # killed mid-write must not leave a truncated mp3 that every later
        # request then serves as a cache hit.
        temporary = path.with_suffix(".mp3.part")
        temporary.write_bytes(audio)
        temporary.replace(path)
        _commit_volume()
    except Exception as err:  # noqa: BLE001 - the audio is already generated
        logger.warning("tts: could not cache %s: %s", key, err)

    logger.info("tts: synthesised %d chars in %.2fs (%d bytes)", len(text), took, len(audio))
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"X-Cache": "miss", "X-Cache-Key": key, "X-Synthesis-Seconds": f"{took:.2f}"},
    )


def set_volume(volume) -> None:
    """Hands this module the *mounted* cache Volume.

    Called from modal_app.py inside the container, because a Volume has to be
    committed through the object the function was given — a fresh
    `Volume.from_name()` handle is not the mounted one. Left unset when
    running locally, where the cache directory is just a directory.
    """
    global _volume
    _volume = volume


def _commit_volume() -> None:
    """Makes a just-written file visible to other containers.

    Modal Volumes are explicitly committed, not automatically: "you need to
    explicitly commit any changes you make to the volume for the changes to
    become visible outside the current container" (modal.Volume's own docs).
    Without this the cache would only help the container that happened to
    synthesise the line, which for a scale-to-zero service is almost never.

    Note the mirror image of this, which is deliberately NOT done: a
    long-lived container does not see other containers' commits without
    reload(), and reload() both costs a network round trip per request and
    errors while any file on the volume is open. The cost of skipping it is a
    duplicate synthesis, which the next commit then de-duplicates.
    """
    if _volume is None:
        return
    try:
        _volume.commit()
    except Exception as err:  # noqa: BLE001 - the audio is already in hand
        logger.warning("tts: volume commit failed: %s", err)
