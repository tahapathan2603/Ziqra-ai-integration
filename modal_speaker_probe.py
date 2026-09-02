"""
Which of this model's English speakers should be the interviewer.

    modal run modal_speaker_probe.py

`ai4bharat/indic-parler-tts` ships 21 English voices, all of them Indian
English speakers — that is what the model is — and recommends two, Thoma and
Mary. Picking between them by ear is not something this environment can do, so
each candidate is measured instead, on the two things that actually matter for
an interviewer's voice:

  intelligibility   the clip is transcribed by the deployed pipeline (Whisper
                    large-v3) and compared word-for-word with the line it was
                    asked to say. A voice the app's own ASR mishears is a
                    voice candidates will mishear.
  Indian English    the same pipeline reports mother-tongue influence and
                    pronunciation, fitted to human ratings. A voice meant to
                    sound like an Indian interviewer should read as one here.

Also reported: seconds per clip and audio duration, so the speed work has a
number, and whether two different lines from one speaker come back with the
same character — a voice that moves between questions is the bug this is
meant to prevent.

Generation runs on the TTS container; scoring goes through the deployed
/extract, called from inside Modal so the client token never leaves it.
"""

import json
import time

import modal

from modal_app import HF_SECRET_NAME, MODAL_SECRET_NAME, tts_cache, tts_image

app = modal.App(
    "ziqra-speaker-probe",
    # No pip_install here: tts_image already ends with add_local_python_source,
    # and Modal refuses a build step after that ("An image tried to run a
    # build step after using image.add_local_*"). The extract call below uses
    # urllib instead of adding a dependency to the deployed image for a
    # probe's sake.
    image=tts_image.add_local_python_source("modal_app", "modal_speaker_probe"),
)

REPORT_PATH = "/cache/tts/speaker-report.json"

EXTRACT_URL = "https://ramramjibkn--ziqra-audio-api-fastapi-app.modal.run"

# The two the model card recommends for English, plus four more from its
# English set for contrast.
# Round two: the two that measured best, each with the style prompt that asks
# for a measured pace and one that does not. Speaker choice moved audio
# duration by a factor of nearly two for identical text (Mary 4.4s vs Jatin
# 8.3s), and generation tracks duration at about 1.2x real time — so the words
# in this description are a speed setting, not just a stylistic one.
CANDIDATES = ["Mary", "Thoma"]

LINES = [
    "Tell me about a project you are proud of, and what your part in it was.",
    "What would you do differently if you faced that same situation again?",
]

STYLE = (
    "speaks Indian English in a clear, warm, professional tone at a measured pace, as if "
    "interviewing someone. The recording is very close-sounding and completely noise-free."
)
STYLE_BRISK = (
    "speaks Indian English in a clear, warm, professional tone, as if interviewing someone. "
    "The recording is very close-sounding and completely noise-free."
)
STYLES = {"measured": STYLE, "brisk": STYLE_BRISK}


def _find_transcript(node, depth: int = 0):
    """Finds the transcript wherever the pipeline puts it.

    The first run of this probe reported WER 1.0 for every speaker with an
    empty transcript, while pronunciation_score came back with real values —
    so the audio was fine and the *path* was wrong. Rather than guess again,
    walk the response for the first plausible transcript field.
    """
    if depth > 4 or not isinstance(node, (dict, list)):
        return None
    if isinstance(node, list):
        parts = [p for p in (_find_transcript(item, depth + 1) for item in node) if p]
        return " ".join(parts) or None
    for key in ("transcript", "text", "full_transcript", "transcription"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in node.values():
        found = _find_transcript(value, depth + 1)
        if found:
            return found
    return None


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={"/cache": tts_cache},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME), modal.Secret.from_name(HF_SECRET_NAME)],
)
def probe() -> str:
    """Generates every line for every candidate, reads each back through the
    deployed pipeline, and writes the report to the volume."""
    import os
    import urllib.request
    import uuid
    from pathlib import Path

    from backend.api import tts

    tts.set_volume(tts_cache)
    tts._load()
    if tts._model is None:
        Path(REPORT_PATH).write_text(json.dumps({"error": tts._load_error}))
        tts_cache.commit()
        return REPORT_PATH

    def words(text: str) -> list:
        return [w for w in "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split() if w]

    def wer(reference: str, hypothesis: str) -> float:
        """Levenshtein over words, normalised — the standard measure."""
        ref, hyp = words(reference), words(hypothesis)
        if not ref:
            return 1.0
        previous = list(range(len(hyp) + 1))
        for i, r in enumerate(ref, 1):
            current = [i]
            for j, h in enumerate(hyp, 1):
                current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
            previous = current
        return round(previous[-1] / len(ref), 3)

    # The first generation pays for compilation, so it is reported separately
    # rather than charged to a speaker.
    warm_started = time.time()
    tts.synthesise("Ready.", f"Thoma {STYLE}")
    report = {"compile_and_first_generation_seconds": round(time.time() - warm_started, 2), "speakers": {}}

    token = os.environ["CLIENT_TOKEN"]
    for speaker, (style_name, style) in [(s, st) for s in CANDIDATES for st in STYLES.items()]:
        voice = f"{speaker} {style}"
        rows = report["speakers"].setdefault(f"{speaker}/{style_name}", [])
        for line in LINES:
            started = time.time()
            audio = tts.synthesise(line, voice)
            took = round(time.time() - started, 2)

            row = {"line": line, "generate_seconds": took, "bytes": len(audio)}
            try:
                # Multipart by hand, because urllib has no equivalent of
                # requests' files= and this container has no requests.
                boundary = uuid.uuid4().hex
                body_bytes = b"".join(
                    [
                        f"--{boundary}\r\n".encode(),
                        f'Content-Disposition: form-data; name="file"; filename="{speaker}.mp3"\r\n'.encode(),
                        b"Content-Type: audio/mpeg\r\n\r\n",
                        audio,
                        f"\r\n--{boundary}--\r\n".encode(),
                    ]
                )
                request = urllib.request.Request(
                    f"{EXTRACT_URL}/extract",
                    data=body_bytes,
                    headers={
                        "X-Client-Token": token,
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                    },
                )
                with urllib.request.urlopen(request, timeout=900) as handle:
                    body = json.loads(handle.read().decode("utf-8"))
                analysis = (body.get("level2") or {}).get("analysis") or {}
                level1 = body.get("level1") or {}
                transcript = _find_transcript(level1) or _find_transcript(body) or ""
                mti = analysis.get("mti") or {}
                row["level1_keys"] = sorted(level1.keys())
                row.update(
                    {
                        "transcript": transcript,
                        "wer": wer(line, transcript),
                        "audio_seconds": level1.get("duration") or (body.get("meta") or {}).get("duration_seconds"),
                        "pronunciation_score": (analysis.get("pronunciation") or {}).get("pronunciation_score"),
                        "mti": {k: v for k, v in mti.items() if not isinstance(v, (list, dict))},
                    }
                )
            except Exception as err:  # noqa: BLE001 - recorded, not raised
                row["error"] = f"{type(err).__name__}: {err}"
            rows.append(row)

    Path(REPORT_PATH).write_text(json.dumps(report, indent=2, default=str))
    tts_cache.commit()
    return REPORT_PATH


@app.local_entrypoint()
def main() -> None:
    print("report written to", probe.remote(), "in volume ziqra-tts-cache")
