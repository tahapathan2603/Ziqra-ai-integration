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

# The speaker is chosen by NAMING one, then describing the delivery — that is
# Parler's whole interface.
#
# The name has to come from this model's own English set, and that is what was
# wrong before: it said "Anjali", who is a speaker for other languages in this
# model but NOT one of its English voices, so there was no speaker to lock on
# to and every generation sampled a different voice. The accent audibly moved
# between questions for exactly that reason. The model card lists 21 English
# speakers and recommends two of them, Thoma and Mary; every one of them is an
# Indian English voice, since that is what this model is.
#
# Which of them is used is decided by measurement, not taste — see
# modal_speaker_probe.py, which reads each candidate back through this repo's
# own pipeline and compares transcription accuracy and mother-tongue
# influence. Overridable so that choice can change without a code deploy.
# Measured, not chosen by ear: modal_speaker_probe.py generates the same two
# interview lines for six of this model's English speakers and reads each back
# through this repo's own fitted pronunciation model. Mary won on both axes at
# once — 100 and 91 against Thoma's 98 and 87, and the fastest of the six
# because she produces the shortest audio for identical text (4.4s where Jatin
# took 8.3s), and generation tracks audio duration at about 1.2x real time.
VOICE_SPEAKER = os.environ.get("TTS_VOICE_SPEAKER", "Mary")

# The wording follows the model card rather than reading nicely:
#
#   * "very clear audio" is a documented magic phrase — "Include the term
#     'very clear audio' to generate the highest quality audio".
#   * speaking rate, pitch, background noise and reverberation are all
#     controlled through this description, so each is stated explicitly
#     instead of left to the model.
#   * "moderate speaking rate", not "a measured pace": generation time tracks
#     audio duration almost linearly, so this phrase is a speed setting as
#     much as a stylistic one.
VOICE_STYLE = os.environ.get(
    "TTS_VOICE_STYLE",
    "speaks in a clear, warm and professional tone at a moderate speaking rate with balanced pitch, "
    "as if interviewing someone. Very clear audio, very close-sounding, with no background noise.",
)
VOICE = os.environ.get("TTS_VOICE", f"{VOICE_SPEAKER} {VOICE_STYLE}")

VOICE_SEED = int(os.environ.get("TTS_VOICE_SEED", "42"))

# Padding both tokenizers to fixed lengths is what makes torch.compile pay:
# it keeps the input shapes static, so the compiled graph is reused instead of
# recompiled per line length (parler-tts's own INFERENCE.md guidance).
DESCRIPTION_TOKENS = 64
PROMPT_TOKENS = 64

# A ceiling on generated audio, in decoder tokens. Long enough for any
# interview line at ~86 tokens/second of audio, and it stops a pathological
# generation from running away with the GPU.
MAX_NEW_TOKENS = int(os.environ.get("TTS_MAX_NEW_TOKENS", "1500"))


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
            # float16 on the GPU, not the checkpoint's float32. This is the
            # single biggest win available and it was simply being left on the
            # table: the A10G has fp16 tensor cores and this model is a
            # transformer decoder, so it is the arithmetic that dominates.
            dtype = torch.float16 if device == "cuda" else torch.float32
            # No attn_implementation="sdpa" here, though parler-tts's guide
            # suggests it: measured, it refuses to load at all —
            # "T5EncoderModel does not support an attention implementation
            # through torch.nn.functional.scaled_dot_product_attention yet"
            # (transformers 4.46). The setting applies to the whole model, and
            # this one contains a T5 text encoder, so the default stands and
            # the speed comes from fp16 and the compiled decoder below.
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            ).to(device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            # Parler reads the speaker description with the text encoder's own
            # tokenizer, which is a different one from the prompt tokenizer.
            description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
            _model, _tokenizer, _description_tokenizer = model, tokenizer, description_tokenizer
            logger.info("tts: loaded %s on %s (%s) in %.1fs", MODEL_ID, device, dtype, time.time() - started)

            # No static cache and no torch.compile, though parler-tts's
            # INFERENCE.md recommends both — measured, they do not work on
            # this version pair and the failure is at generate() time, not at
            # set-up time, so it would have taken the whole voice down:
            #
            #   AttributeError: 'StaticCache' object has no attribute
            #   'max_batch_size'
            #
            # parler-tts reads that attribute; transformers 4.46 renamed it to
            # `batch_size` (its own deprecation notice says so). Pinning a
            # transformers old enough for parler and new enough for the model
            # is a bigger change than the win justifies, and without a static
            # cache torch.compile only recompiles per line length. fp16 above
            # is where the speed comes from; the caches and the app's prefetch
            # are where the *felt* speed comes from.
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
    # padding to a fixed length, not to the batch: static shapes are what let
    # the compiled graph above be reused instead of recompiled for every
    # different line length.
    description_ids = _description_tokenizer(
        voice, return_tensors="pt", padding="max_length", max_length=DESCRIPTION_TOKENS, truncation=True
    ).to(device)
    prompt_ids = _tokenizer(
        text, return_tensors="pt", padding="max_length", max_length=PROMPT_TOKENS, truncation=True
    ).to(device)
    # Deterministic: the same line is the same voice every time, and the voice
    # does not drift between questions.
    torch.manual_seed(VOICE_SEED)
    with torch.inference_mode():
        generation = _model.generate(
            input_ids=description_ids.input_ids,
            attention_mask=description_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    audio = generation.to(torch.float32).cpu().numpy().squeeze()
    return _to_mp3(audio, _model.config.sampling_rate)


@app.on_event("startup")
def _begin_warmup() -> None:
    """Loads the model and pays the compile cost before any request arrives.

    In a thread, not inline: the server has to start answering /health
    immediately, and the Worker's readiness check depends on that. The first
    generation is what actually triggers compilation, so a throwaway line is
    synthesised here — otherwise the first candidate to ask a question pays a
    minute of it.
    """

    def run() -> None:
        try:
            _load()
            if _model is None:
                return
            started = time.time()
            synthesise("Ready.", VOICE)
            logger.info("tts: warm-up generation took %.1fs", time.time() - started)
        except Exception as err:  # noqa: BLE001 - a failed warm-up is not fatal
            logger.warning("tts: warm-up failed: %s", err)

    threading.Thread(target=run, daemon=True, name="tts-warmup").start()


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
