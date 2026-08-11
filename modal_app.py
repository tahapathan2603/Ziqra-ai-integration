"""
Modal deployment for the audio feature-extraction API.

Infrastructure only -- no pipeline or feature-extraction logic lives here.
This file's only job is to get backend/api/main.py's FastAPI `app` running
on a Modal GPU container and serve it over HTTP; the app object itself
(routes, schemas, error handling) is defined once, in backend/api/main.py,
and reused unchanged for both local and Modal runs.

Local run (no Modal -- for comparison/debugging):
    uvicorn backend.api.main:app --reload --port 8000

Modal:
    pip install modal
    modal setup                    # one-time browser auth
    modal serve modal_app.py       # dev: hot-reload, temporary URL, streams logs
    modal deploy modal_app.py      # persistent URL
"""

from pathlib import Path

import modal


MODAL_SECRET_NAME = "custom-secret"

# requirements.txt is shared with local dev, which needs gradio for
# backend/preprocessing/vad_ui.py's dev-only Gradio UI. The Modal container
# never imports gradio (confirmed: it appears nowhere in backend/ outside
# vad_ui.py), so it has no business being installed there -- and installing
# it anyway actively BREAKS the build: gradio 6.20.0 requires
# huggingface-hub>=1.2.0, but huggingface_hub is pinned to 0.36.2 in
# requirements.txt for the actual pipeline's ML stack (transformers/
# accelerate/faster-whisper). `pip install -r requirements.txt` from a
# clean image can't satisfy both floors at once and fails outright --
# confirmed to already be a real, if silent, inconsistency even in local
# dev (`pip check` flags the exact same conflict there; it "works" locally
# only because gradio's own huggingface-hub floor is never actually
# exercised in dev usage).
#
# Bumping huggingface_hub's pin instead was deliberately rejected: that
# package is imported throughout the real pipeline (transformers,
# accelerate, faster-whisper all depend on it), so a version it wasn't
# validated against is a correctness risk to the thing this deployment
# actually needs to work -- for the sake of a UI framework it doesn't even
# use. Excluding gradio carries none of that risk and also shrinks the
# image (gradio pulls in a large, otherwise-unused dependency tree).
#
# Confirmed narrowly scoped to just these five packages -- not the
# transformers/torch/etc. stack: gradio_client's own huggingface-hub floor
# (>=0.19.3) is already satisfied by 0.36.2, so it isn't part of the
# conflict; it (and hf-gradio/safehttpx/groovy) are excluded anyway because
# none of the four are imported anywhere outside gradio's own UI stack.
#
# `modal` itself is excluded for a different, equally confirmed reason: it
# is never imported anywhere under backend/ (nothing that runs INSIDE the
# container needs it -- only modal_app.py/modal_test.py, which run locally
# to define the deployment). It also isn't optional to exclude the way
# gradio was a choice -- installing it breaks the build outright: modal
# 1.5.3 requires protobuf<7.0, but requirements.txt pins protobuf==7.35.1
# for opentelemetry-proto's floor (>=5.0) elsewhere in this same file.
# Checked directly in modal's own Image.debian_slim source before relying
# on this: "# after 2024.10, modal requirements are mounted at runtime" --
# Modal supplies its own client/runtime into every container through its
# own mechanism, independent of anything pip_install_from_requirements
# does, so there was never a reason for the container's own requirements to
# list it, whether or not the protobuf conflict existed.
#
# `torchao` is excluded for the same shape of reason as `modal`: nothing
# under backend/ imports it except backend/knowledge_distillation/ (the
# QLoRA TRAINING path -- torchao==0.18.0 is pinned there for PEFT's LoRA
# dispatcher on Kaggle T4s, per that code's own history; entirely unrelated
# to this container, which only ever runs feature extraction). Its presence
# actively broke Wav2Vec2 loading on Modal: torchao 0.18.0's own top-level
# code does `from torch.nn.functional import ScalingType`, a symbol that
# doesn't exist before torch 2.11 -- but requirements.txt pins torch==2.8.0
# (chosen for its cu128 CUDA build; bumping it was rejected as a fix, see
# below). Confirmed the exact failure mechanism before excluding, not just
# reasoned about it: transformers/quantizers/auto.py imports
# TorchAoHfQuantizer at MODULE TOP LEVEL -- unconditionally, on every
# `transformers` import, regardless of whether quantization is requested --
# which runs quantizer_torchao.py's `if is_torchao_available(): import
# torchao`. That guard is satisfied by metadata inspection alone
# (importlib.metadata.version("torchao"), confirmed by reading
# transformers' own _is_package_available -- it never actually imports the
# package to check), so a torchao that is installed-but-broken passes the
# guard as "available" and then detonates for real on the unguarded `import
# torchao` right after. Excluding it here means that probe correctly finds
# torchao absent and skips it -- exactly what already happens in local dev,
# where torchao was never installed and Wav2Vec2 has loaded correctly all
# session. Upgrading torch to >=2.11 instead was rejected: it would touch
# the pinned cu128 CUDA build and its torchaudio/torchvision pairing for a
# package (torchao) this container never uses, i.e. strictly more risk for
# strictly less benefit than not installing it. Confirmed safe to drop:
# neither peft, accelerate, nor transformers declare a hard (non-extras)
# dependency on torchao.
#
# Hyphenated, not underscored, for "gradio-client" -- must match the
# normalized form the filter below compares against (pip package names
# treat "_" and "-" as interchangeable; requirements.txt spells this one
# "gradio_client", so the comparison normalizes both sides the same way
# before matching).
_CONTAINER_EXCLUDED_PACKAGES = {"gradio", "gradio-client", "hf-gradio", "safehttpx", "groovy", "modal", "torchao"}


MODAL_REQUIREMENTS_FILE = "requirements-modal.txt"


def _generate_modal_requirements() -> None:
    """
    Regenerates requirements-modal.txt from requirements.txt, dropping the
    lines this container doesn't need (see _CONTAINER_EXCLUDED_PACKAGES).
    Run this by hand -- `python modal_app.py` -- whenever requirements.txt
    changes; it is NOT called automatically anywhere in this file.

    That "not called automatically" is deliberate, and the fix for a real
    bug: this used to run inline as part of building `image` below (writing
    to a tempfile instead of this checked-in path), on the theory that it
    only ever runs locally, once, when you invoke `modal run`/`modal
    deploy`. That's wrong. Confirmed by a real failure: to execute
    `_prefetch_models` (below) as a build step, Modal has to import this
    entire module INSIDE that build's remote container to get a handle on
    the function -- and importing a module re-runs every line of its
    top-level code, including whatever built `image`. That remote container
    has only what earlier image layers baked into it; your actual project
    tree -- requirements.txt included -- was never copied there, so a local
    Path(...).read_text() call sitting at module top level failed with
    FileNotFoundError the moment the container tried to import the module.

    requirements-modal.txt sidesteps this completely: it's a real file
    checked into the repo, so `image` below only ever references it by a
    plain string -- no function call, no disk I/O in code that runs at
    import time, so re-importing this module remotely has nothing local
    left to fail on. (Whatever file reading pip_install_from_requirements
    itself needs happens inside Modal's own lazy build machinery, which
    already builds and caches this exact layer successfully -- confirmed by
    it completing before this bug's crash, in the next layer over.)
    """
    lines = Path("requirements.txt").read_text().splitlines()
    kept = [
        line for line in lines
        if not line.strip()
        or line.strip().startswith("#")
        or line.split("==")[0].strip().lower().replace("_", "-") not in _CONTAINER_EXCLUDED_PACKAGES
    ]
    header = [
        "# GENERATED FILE -- do not hand-edit.",
        f"# Regenerate with: python {Path(__file__).name}",
        "# Derived from requirements.txt by dropping packages this container",
        "# doesn't need -- see modal_app.py's _CONTAINER_EXCLUDED_PACKAGES for why.",
        "",
    ]
    Path(MODAL_REQUIREMENTS_FILE).write_text("\n".join(header + kept) + "\n")


def _prefetch_models() -> None:
    """
    Runs once at image-BUILD time (see image.run_function below), not at
    request time: downloads and caches the two Hugging-Face-hosted models
    the pipeline loads lazily on first use, so they're baked into the image
    layer instead of re-downloading several GB on every cold container
    start. (Silero VAD needs no prefetch -- its model ships inside the
    `silero_vad` PyPI package itself, no network call at load time.)

    Also SMOKE-TESTS the two apt-installed system dependencies. Without
    this, a missing/undiscoverable ffmpeg or espeak-ng wouldn't surface
    until an actual extraction request, minutes into a GPU run, as an
    obscure failure deep inside the pipeline -- and it would fail that way
    for every request, forever, on a "successfully built" image. Checking
    here converts that into a loud, immediate build failure. Both are
    genuinely fragile in a way `apt_install` alone doesn't guarantee:
    phonemizer doesn't invoke the espeak-ng *binary*, it dlopen()s
    libespeak-ng.so through ctypes, so the binary being installed is not
    by itself proof the library is discoverable.

    Everything here is imported from pip/apt packages only, never from
    backend.* -- local source is deliberately added AFTER this build step
    (see the image chain below), so backend.* is not importable yet at this
    point. Keep the model identifiers in sync with their real definitions:
      - backend/preprocessing/speech_to_text.py's MODEL_SIZE
      - backend/feature_extractors/audio/pronunciation/phoneme_accuracy.py's
        MODEL_NAME
    """
    import subprocess

    from faster_whisper import WhisperModel
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    # ffmpeg: silero_vad._convert_to_wav shells out to it for every non-wav upload.
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)

    # espeak-ng via phonemizer, exercised the same way the pipeline uses it
    # (phoneme_accuracy.get_expected_phonemes) rather than just checking the
    # binary exists -- this is the call that actually dlopen()s the library.
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    probe = phonemize(
        ["hello"], language="en-us", backend="espeak",
        separator=Separator(phone=" ", word=""), strip=True, njobs=1,
    )
    if not probe or not probe[0].strip():
        raise RuntimeError(
            "espeak-ng/phonemizer produced no phonemes for a known word at image-build "
            "time. The pronunciation analyzer would silently degrade for every request. "
            "Check that apt_install('espeak-ng') still provides a discoverable "
            "libespeak-ng shared library in this base image."
        )

    WhisperModel("large-v3", device="cpu", compute_type="int8")
    Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft", low_cpu_mem_usage=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    # System packages the pipeline shells out to / depends on at *runtime*,
    # not just build time -- both are hard requirements, not conveniences:
    #   ffmpeg     backend/preprocessing/silero_vad.py's _convert_to_wav
    #              shells out to it directly for any non-wav upload.
    #   espeak-ng  backend/feature_extractors/audio/pronunciation/
    #              phoneme_accuracy.py calls phonemizer with
    #              backend="espeak", which wraps this binary.
    .apt_install("ffmpeg", "espeak-ng")
    # A plain string, not a function call, deliberately -- see
    # _generate_modal_requirements's docstring for why anything that reads
    # a local file at module top level breaks the moment Modal re-imports
    # this module inside a remote build-step container.
    .pip_install_from_requirements(MODAL_REQUIREMENTS_FILE)
    .run_function(_prefetch_models)
    # Added LAST, deliberately: local source is what changes on every
    # iteration during development, while the two layers above (apt/pip
    # installs, multi-GB model downloads) are expensive and should stay
    # cached across those changes. If this were placed earlier, editing any
    # backend/*.py file would invalidate _prefetch_models too, and every
    # code change would re-download both models on the next build.
    .add_local_python_source("backend")
)

app = modal.App("ziqra-audio-api", image=image)


@app.function(
    gpu="A10G",
    # Raised from 600s. A cold container has to import torch/transformers,
    # load Whisper large-v3 onto the GPU, and load wav2vec2 -- all BEFORE
    # any audio is processed, and all inside this budget. 600s left little
    # headroom for a long recording that landed on a cold start; the cost
    # of a too-low value is a killed request after minutes of real work,
    # while the cost of a too-high one is only that a genuinely wedged
    # request holds its GPU longer before being reaped.
    timeout=1200,
    scaledown_window=300,  # keep the container (and its loaded models) warm for 5 min after the last request
    # Cost guard for the testing phase: without a ceiling, a burst of
    # requests autoscales GPU containers with nothing to bound the bill.
    # Each container is also a multi-GB cold start, so unbounded fan-out
    # is not even a latency win here.
    max_containers=2,
    # Concurrency is deliberately left at Modal's default of ONE input per
    # container -- verified, not assumed: `modal.concurrent` is opt-in, and
    # its own docs note that for synchronous functions it uses
    # multi-threading and "the code must be thread-safe". This pipeline's
    # models are module-level singletons (speech_to_text._model,
    # phoneme_accuracy._model) shared process-wide and never verified
    # thread-safe, and analyze_audio already parallelizes internally. Adding
    # @modal.concurrent here would put concurrent requests through those
    # shared models, so it must not be enabled without testing that first.
    #
    # Injects every key/value in the named Secret as an environment
    # variable in the container -- CLIENT_TOKEN ends up in os.environ
    # exactly as if it had been `export`ed, which is what
    # verify_client_token() (backend/api/main.py) reads. The token value
    # itself lives only in Modal's secret store, never in this repo.
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
@modal.asgi_app()
def fastapi_app():
    # Imported inside the function body, not at module top level: this file
    # is executed locally by the `modal` CLI to define the App/Image, and
    # the local machine has neither `backend`'s heavy ML dependencies
    # installed nor any need to import them just to describe the
    # deployment -- only the remote container needs this import to
    # succeed, and it does, since `backend` is part of the image (see
    # add_local_python_source above) and MODAL_REQUIREMENTS_FILE is installed.
    from backend.api.main import app as web_app

    return web_app


if __name__ == "__main__":
    _generate_modal_requirements()
    print(f"Wrote {MODAL_REQUIREMENTS_FILE}")
