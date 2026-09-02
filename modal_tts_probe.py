"""
Probe for the Indian-English TTS voice, in the same container it serves from.

    modal run modal_tts_probe.py

Calls backend/api/tts.py's synthesis directly rather than over HTTP, so it
answers the two questions the HTTP endpoint cannot while the shared client
token lives only in a Modal Secret:

  1. can this container get the gated model at all, and if not, what exactly
     does Hugging Face say
  2. how long does a line take, cold and warm — the number that decides
     whether Modal TTS is usable inside the interview loop or only for
     pre-cached questions

Prints, never raises: a gated-model refusal is a result here, not a crash.
"""

import time

import modal

from modal_app import MODAL_SECRET_NAME, tts_cache, tts_image

# Its own App, like the other probes here, so running this never touches the
# deployed one. The two local modules have to be added to the image
# explicitly: the container imports this file to find `probe`, and this file
# imports modal_app for the image and volume above — only `backend` comes with
# tts_image.
app = modal.App(
    "ziqra-tts-probe",
    image=tts_image.add_local_python_source("modal_app", "modal_tts_probe"),
)

LINES = [
    "Tell me about yourself.",
    "Describe a project you are proud of and what your part in it was.",
]


@app.function(
    gpu="A10G",
    timeout=1800,
    volumes={"/cache": tts_cache},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
def probe() -> dict:
    import os

    from backend.api import tts

    tts.set_volume(tts_cache)
    report: dict = {"model": tts.MODEL_ID, "hf_token_present": bool(os.environ.get("HF_TOKEN"))}

    started = time.time()
    tts._load()
    report["load_seconds"] = round(time.time() - started, 1)
    report["load_error"] = tts._load_error
    if tts._model is None:
        return report

    report["sampling_rate"] = tts._model.config.sampling_rate
    timings = []
    for index, line in enumerate(LINES):
        began = time.time()
        try:
            audio = tts.synthesise(line, tts.VOICE)
            timings.append(
                {
                    "chars": len(line),
                    "seconds": round(time.time() - began, 2),
                    "bytes": len(audio),
                    # First call also pays CUDA kernel JIT, so it is reported
                    # separately from the ones after it.
                    "first": index == 0,
                }
            )
        except Exception as err:  # noqa: BLE001 - the point is to report it
            timings.append({"chars": len(line), "error": f"{type(err).__name__}: {err}"})
    report["synthesis"] = timings
    return report


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(probe.remote(), indent=2))
