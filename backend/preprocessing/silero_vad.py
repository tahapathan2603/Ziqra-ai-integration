"""Silero VAD preprocessing: detect and isolate speech regions in interview audio."""

import logging
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import torch

# This file is named silero_vad.py, same as the installed `silero_vad` pip package.
# Running it directly (`python preprocessing/silero_vad.py`) makes Python auto-insert
# this file's own directory at sys.path[0], which would shadow the real package on
# the next import line. Push it to the end of sys.path instead, so the package
# resolves correctly. No-op when this module is imported normally.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _THIS_DIR:
    sys.path.append(sys.path.pop(0))

# Also make the project root importable, so this file's __main__ block can reach
# sibling top-level packages (e.g. backend.feature_extractors) via proper dotted
# imports when run directly (`python preprocessing/silero_vad.py` from backend/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio, save_audio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Must stay a superset of backend/api/main.py's ALLOWED_EXTENSIONS — that's
# the API layer's advertised contract, but this check runs first and used to
# be narrower (missing .flac/.webm), so a request main.py accepted still
# blew up here as an unhandled ValueError -> vague 500 instead of the 400 the
# API boundary implies. _convert_to_wav shells out to ffmpeg, which decodes
# any of these containers the same way, so there's no real reason for this
# list to be narrower than main.py's at all.
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".opus", ".ogg", ".flac", ".webm")
CHUNKS_DIR = os.path.join(_THIS_DIR, "..", "storage", "chunks")

_model = None


def _get_model():
    """Load and cache the Silero VAD model (loaded once per process)."""
    global _model
    if _model is None:
        logger.info("Loading Silero VAD model...")
        _model = load_silero_vad()
    return _model


def _convert_to_wav(src_path: str, dst_path: str) -> None:
    """
    Normalize any supported input (wav/mp3/m4a) to 16kHz mono PCM wav via ffmpeg.

    torchaudio's optional decoding backends (sox, torchcodec) are inconsistently
    available across environments and don't reliably cover formats like m4a/AAC,
    so we shell out to ffmpeg directly rather than depend on them.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-vn",
        "-loglevel", "error",
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode '{src_path}': {result.stderr.strip()}")


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """Load an audio file (.wav/.mp3/.m4a) as a mono waveform resampled to 16kHz."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    logger.info(f"Loading audio: {path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav_path = tmp.name

    try:
        _convert_to_wav(path, tmp_wav_path)
        waveform = read_audio(tmp_wav_path, sampling_rate=SAMPLE_RATE)
    finally:
        os.remove(tmp_wav_path)

    return waveform, SAMPLE_RATE


def detect_speech_segments(
    waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE
) -> List[Dict[str, float]]:
    """Run Silero VAD on a waveform and return raw speech segments in seconds."""
    logger.info("Running Silero VAD...")
    model = _get_model()

    raw_timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sample_rate,
        return_seconds=True,
    )

    segments = [
        {"start": float(ts["start"]), "end": float(ts["end"])} for ts in raw_timestamps
    ]
    logger.info(f"Detected {len(segments)} speech regions.")
    return segments


def merge_segments(
    segments: List[Dict[str, float]], max_gap: float = 0.5
) -> List[Dict[str, float]]:
    """Merge consecutive speech segments separated by a gap of max_gap seconds or less."""
    if not segments:
        return []

    logger.info("Merging short pauses...")
    ordered = sorted(segments, key=lambda s: s["start"])
    merged = [dict(ordered[0])]

    for segment in ordered[1:]:
        previous = merged[-1]
        gap = segment["start"] - previous["end"]
        if gap <= max_gap:
            previous["end"] = max(previous["end"], segment["end"])
        else:
            merged.append(dict(segment))

    return merged


def create_chunks(
    waveform: torch.Tensor, sample_rate: int, segments: List[Dict[str, float]]
) -> List[torch.Tensor]:
    """Slice the waveform into speech-only audio chunks based on the given segments."""
    total_samples = waveform.shape[-1]
    chunks = []

    for segment in segments:
        start_sample = max(0, int(segment["start"] * sample_rate))
        end_sample = min(total_samples, int(segment["end"] * sample_rate))
        chunks.append(waveform[start_sample:end_sample])

    return chunks


def process_audio(path: str, max_gap: float = 0.5) -> Dict:
    """
    Run the full VAD pipeline: load audio, detect speech, merge short pauses,
    and split into speech chunks.

    Returns a dict with:
        speech_segments: List[{"start": float, "end": float}] (merged, seconds)
        sample_rate: int
        waveform: torch.Tensor (full loaded waveform, mono, 16kHz)
        chunks: List[torch.Tensor] (one waveform slice per merged segment)
    """
    waveform, sample_rate = load_audio(path)
    raw_segments = detect_speech_segments(waveform, sample_rate)
    merged_segments = merge_segments(raw_segments, max_gap=max_gap)
    chunks = create_chunks(waveform, sample_rate, merged_segments)

    return {
        "speech_segments": merged_segments,
        "sample_rate": sample_rate,
        "waveform": waveform,
        "chunks": chunks,
    }


if __name__ == "__main__":
    # Dev-only entry point: launches the Gradio testing UI, kept in a separate
    # sibling module so the VAD logic above has no UI dependency of its own.
    import speech_to_text
    import timestamps
    import vad_ui
    from backend.distillation.dataset_builder import build_and_save_dataset
    from backend.feature_extractors.audio.audio_analyzer import analyze_audio
    from backend.feedback.coach_packets import build_coach_packets
    from backend.feedback.evidence_packet import build_evidence_packet
    from backend.feedback.feedback_generator import generate_feedback

    vad_ui.launch(
        process_audio=process_audio,
        save_audio_fn=save_audio,
        chunks_dir=CHUNKS_DIR,
        process_chunks=speech_to_text.process_chunks,
        stt_model_name=speech_to_text.MODEL_SIZE,
        process_timestamps=timestamps.process_timestamps,
        analyze_audio=analyze_audio,
        build_and_save_dataset=build_and_save_dataset,
        dataset_dir=os.path.join(_PROJECT_ROOT, "dataset"),
        build_coach_packets=build_coach_packets,
        build_evidence_packet=build_evidence_packet,
        generate_feedback=generate_feedback,
    )
