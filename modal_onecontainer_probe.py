"""
Can the transcription pipeline and the TTS voice share one container?

    modal run modal_onecontainer_probe.py

Two services on the Modal dashboard is a fair thing to question, so this
answers it with a build rather than an opinion. It installs the pipeline's own
requirements and then parler-tts on top, reports what pip actually resolved
to, and tries to load *both* models and use them.

The pins disagree on paper:

    transformers   pipeline 4.57.6   parler 4.46.1
    numpy          pipeline 2.4.6    parler <2

pip will happily resolve that by moving one of them, and the interesting
question is which, and whether what is left still works.
"""

import json

import modal

MODAL_REQUIREMENTS_FILE = "requirements-modal.txt"

combined = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng", "git")
    .pip_install_from_requirements(MODAL_REQUIREMENTS_FILE)
    # On top, deliberately: this is the order a merge would have to happen in,
    # so whatever pip does to the pins above is what a merged service would
    # actually be running.
    .pip_install("git+https://github.com/huggingface/parler-tts.git@main")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("backend")
    .add_local_dir(
        "backend/feature_extractors/audio/models",
        remote_path="/root/backend/feature_extractors/audio/models",
    )
    # add_local_python_source carries .py files only, so the sample the
    # pipeline half needs has to be added as data — the same trap the fitted
    # scorer bundle fell into (see modal_app.py).
    .add_local_dir("backend/test_audio", remote_path="/root/backend/test_audio")
)

app = modal.App("ziqra-onecontainer-probe", image=combined)
cache = modal.Volume.from_name("ziqra-tts-cache", create_if_missing=True)
# Written to the volume, not just returned: the local Modal client dropped
# this run twice ("Deadline exceeded", then "'Connection' object has no
# attribute '_transport'") and took the printed result with it each time.
REPORT_PATH = "/cache/fixtures/onecontainer-report.json" 


@app.function(
    gpu="A10G",
    timeout=1800,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("custom-secret"), modal.Secret.from_name("huggingface")],
)
def probe() -> dict:
    import importlib.metadata as md

    report = {}
    for package in ("transformers", "numpy", "torch", "torchaudio", "faster-whisper", "parler_tts"):
        try:
            report[package] = md.version(package)
        except Exception as err:  # noqa: BLE001
            report[package] = f"<{type(err).__name__}>"

    # 1. does the TTS side still work on whatever pip settled on?
    try:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts", torch_dtype=torch.float16
        ).to("cuda")
        tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        d = desc_tok("Mary speaks clearly. Very clear audio.", return_tensors="pt").to("cuda")
        p = tok("Tell me about yourself.", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                input_ids=d.input_ids,
                attention_mask=d.attention_mask,
                prompt_input_ids=p.input_ids,
                prompt_attention_mask=p.attention_mask,
            )
        report["tts"] = {"ok": True, "samples": int(out.shape[-1])}
    except Exception as err:  # noqa: BLE001 - the result is the point
        report["tts"] = {"ok": False, "error": f"{type(err).__name__}: {err}"[:300]}

    # 2. and does the analysis pipeline still work in the same process?
    try:
        from backend.api.pipeline import extract_features

        # NOT chunk_1.wav: that file is 0 bytes in the repo (its "copy" is
        # the real one), and ffmpeg rightly refuses it — which cost this probe
        # a run. This mp3 is one the deployed pipeline has already scored.
        wav = "/root/backend/test_audio/saa_indian_samples/hindi3.mp3"
        import os

        if not os.path.exists(wav):
            report["pipeline"] = {"ok": False, "error": "no sample in image"}
        else:
            import tempfile

            # extract_features(audio_path, chunks_dir) — the caller owns the
            # chunk directory's lifecycle.
            with tempfile.TemporaryDirectory() as chunks:
                result = extract_features(wav, chunks)
            analysis = (result.get("level2") or {}).get("analysis") or {}
            report["pipeline"] = {
                "ok": True,
                "pronunciation": (analysis.get("pronunciation") or {}).get("pronunciation_score"),
            }
    except Exception as err:  # noqa: BLE001
        report["pipeline"] = {"ok": False, "error": f"{type(err).__name__}: {err}"[:300]}

    from pathlib import Path

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text(json.dumps(report, indent=2, default=str))
    cache.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(probe.remote(), indent=2))
