"""
Modal deployment for the Ziqra audio-extraction API.

Infra only — routes/schemas/pipeline logic live in backend/api/main.py and are
served here unchanged via @modal.asgi_app(). Run from anywhere (this file
adds the repo root to sys.path itself):

    modal serve  ziqra_ai_service/app.py   # dev: temp URL, hot reload, streams logs
    modal deploy ziqra_ai_service/app.py   # persistent URL
    python ziqra_ai_service/app.py         # regenerate requirements-modal.txt

Local, no Modal: uvicorn backend.api.main:app --reload --port 8000
"""

import sys
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))  # so "backend" resolves no matter what cwd `modal` was invoked from

MODAL_SECRET_NAME = "custom-secret"
REQUIREMENTS_SRC = _ROOT / "requirements.txt"
REQUIREMENTS_MODAL = Path(__file__).parent / "requirements-modal.txt"

# requirements.txt is shared with local dev and carries packages this
# container must not install: gradio's huggingface-hub floor conflicts with
# the pinned ML stack's; `modal` is never imported inside the container
# (Modal injects its own runtime) and its own protobuf pin conflicts with
# opentelemetry's; `torchao` is training-only and, if present, actively
# breaks Wav2Vec2 loading here (transformers probes its metadata, decides
# it's "available", then a real import blows up on this torch version).
_EXCLUDED = {"gradio", "gradio-client", "hf-gradio", "safehttpx", "groovy", "modal", "torchao"}


def _write_modal_requirements() -> None:
    """Regenerate requirements-modal.txt from requirements.txt. Run by hand
    (`python ziqra_ai_service/app.py`) whenever requirements.txt changes —
    kept as a checked-in file, not generated inline while building `image`,
    because Modal re-imports this whole module inside the remote build
    container to run `_prefetch_models`; a local disk read at import time
    would fail there since the project tree isn't copied in yet."""
    lines = REQUIREMENTS_SRC.read_text().splitlines()
    kept = [
        line for line in lines
        if not line.strip()
        or line.strip().startswith("#")
        or line.split("==")[0].strip().lower().replace("_", "-") not in _EXCLUDED
    ]
    REQUIREMENTS_MODAL.write_text(
        f"# GENERATED — regenerate with: python ziqra_ai_service/app.py\n" + "\n".join(kept) + "\n"
    )


def _prefetch_models() -> None:
    """Build-time only: bakes the two Hugging-Face models into the image so
    cold starts don't re-download several GB, and smoke-tests ffmpeg /
    espeak-ng — both fail silently at request time otherwise (phonemizer
    dlopen()s libespeak-ng rather than shelling out, so `apt_install` alone
    doesn't prove it's discoverable). Keep these ids in sync with
    backend/preprocessing/speech_to_text.py and
    backend/feature_extractors/audio/pronunciation/phoneme_accuracy.py.
    """
    import subprocess

    from faster_whisper import WhisperModel
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)

    probe = phonemize(
        ["hello"], language="en-us", backend="espeak",
        separator=Separator(phone=" ", word=""), strip=True, njobs=1,
    )
    if not probe or not probe[0].strip():
        raise RuntimeError("espeak-ng produced no phonemes at build time — check libespeak-ng is discoverable.")

    WhisperModel("large-v3", device="cpu", compute_type="int8")
    Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft", low_cpu_mem_usage=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng")  # silero_vad's decoder + phonemizer's espeak backend — both runtime deps
    .pip_install_from_requirements(str(REQUIREMENTS_MODAL))
    .run_function(_prefetch_models)
    .add_local_python_source("backend")  # added last: keeps the expensive layers above cached across code edits
)

app = modal.App("ziqra-audio-api", image=image)


@app.function(
    gpu="A10G",
    timeout=1200,  # covers a cold import + Whisper-large-v3 + wav2vec2 load, all before any audio is touched
    scaledown_window=30,  # keep a warm container for 30s after the last request, then scale to zero
    max_containers=2,  # cost guard — each container is a multi-GB cold start, unbounded fan-out isn't a latency win
    enable_memory_snapshot=True,  # snapshots post-import state so repeat cold starts skip re-importing torch/transformers
    # Concurrency stays at Modal's default (1 input/container): the pipeline's
    # models are process-wide singletons never verified thread-safe.
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],  # injects CLIENT_TOKEN, read by backend/api/main.py
)
@modal.asgi_app()
def fastapi_app():
    # Imported inside the function body: this module also runs locally (the
    # `modal` CLI parses it to build the image) where backend's ML deps
    # aren't installed and don't need to be.
    from backend.api.main import app as web_app

    return web_app


if __name__ == "__main__":
    _write_modal_requirements()
    print(f"Wrote {REQUIREMENTS_MODAL}")
