"""
Orchestrates the existing VAD -> STT -> timestamps -> audio-analysis pipeline
for one audio file and returns its Level 1 + Level 2 output as plain dicts.

This is the same six-call chain backend/preprocessing/vad_ui.py's
run_pipeline() already drives (see that function's docstring for the full
picture, including the coach-packet/dataset-persistence steps this module
deliberately omits — this is extraction only, no scoring/feedback):

    process_audio(audio_path)                    -- VAD: speech segments/chunks
        -> save_audio(...) per chunk               -- chunk WAVs, needed because
                                                        process_chunks transcribes
                                                        from file paths, not tensors
    process_chunks(chunk_records, ...)            -- STT: transcript + words
    process_timestamps(stt_result, ...)           -- sentence/word timestamps
    analyze_audio(...)                            -- Fluency/Pronunciation/MTI/
                                                        Intonation/Engagement
    build_timeline(...)                           -- Level 1 (dataset_builder.py)
    build_features(...)                           -- Level 2 (dataset_builder.py)

Uses the pure, in-memory build_timeline/build_features rather than
build_and_save_dataset -- this module writes nothing to disk (see
dataset_builder.py's own docstring: they're exposed separately so Level 1/2
can be produced without persisting). No coach packets, no logging-handler
capture, no display formatting -- all of that is vad_ui.py-specific and out
of scope here.
"""

import logging
import os
import time
from typing import Dict

from silero_vad import save_audio

from backend.api import reliability
from backend.distillation.dataset_builder import build_features, build_timeline, generate_session_id
from backend.feature_extractors.audio.audio_analyzer import analyze_audio
from backend.preprocessing.silero_vad import process_audio
from backend.preprocessing.speech_to_text import process_chunks
from backend.preprocessing.timestamps import process_timestamps

logger = logging.getLogger(__name__)


class NoSpeechDetectedError(Exception):
    """VAD found zero speech segments in the uploaded audio. Raised before
    any transcription/analysis is attempted -- downstream code (STT, the
    five analyzers) is written for at-least-one-chunk input and hasn't been
    verified safe on an empty transcript, so this is a deliberate early exit
    rather than a guess about what would happen if it fell through."""


def extract_features(audio_path: str, chunks_dir: str) -> Dict:
    """
    Run the full pipeline on one audio file already on disk.

    Args:
        audio_path: path to the input audio (any format ffmpeg can decode --
            see silero_vad._convert_to_wav).
        chunks_dir: directory to write per-segment chunk WAVs into. Callers
            own its lifecycle (this function only writes into it) -- the API
            layer uses a fresh tempfile.TemporaryDirectory() per request so
            concurrent requests never share or race on chunk files, unlike
            vad_ui.py's fixed CHUNKS_DIR (wiped on every run, fine for a
            single-user dev UI, unsafe for concurrent API requests).

    Returns:
        {"session_id": str, "level1": dict, "level2": dict,
         "meta": {"duration_seconds": float, "speech_seconds": float,
                   "processing_seconds": float}}
        level1/level2 are exactly dataset_builder.build_timeline/build_features's
        output -- the same schema already written to dataset/timeline/*.json
        and dataset/features/*.json, just returned instead of saved.

    Raises:
        NoSpeechDetectedError: VAD found no speech in the audio.
        RuntimeError: propagated from ffmpeg decode failure (silero_vad.py)
            on unreadable/corrupt audio -- the API layer maps this to a 400.
    """
    start_time = time.time()

    vad_result = process_audio(audio_path)
    vad_segments = vad_result["speech_segments"]
    chunks = vad_result["chunks"]
    sample_rate = vad_result["sample_rate"]
    waveform = vad_result["waveform"]
    total_duration = waveform.shape[-1] / sample_rate

    if not vad_segments:
        raise NoSpeechDetectedError("No speech segments detected in the uploaded audio.")

    os.makedirs(chunks_dir, exist_ok=True)
    chunk_records = []
    for i, (chunk, seg) in enumerate(zip(chunks, vad_segments), start=1):
        chunk_path = os.path.join(chunks_dir, f"chunk_{i}.wav")
        save_audio(chunk_path, chunk, sampling_rate=sample_rate)
        chunk_records.append({"chunk_id": i, "path": chunk_path, "start": seg["start"], "end": seg["end"]})

    # word_timestamps=True computes word-level timing in this same STT pass,
    # so process_timestamps() below doesn't re-transcribe to get it.
    stt_result = process_chunks(chunk_records, word_timestamps=True)
    ts_result = process_timestamps(stt_result, chunk_records, words=stt_result.get("words"))

    speech_duration = sum(seg["end"] - seg["start"] for seg in vad_segments)
    speech_chunks = [{"chunk_id": r["chunk_id"], "start": r["start"], "end": r["end"]} for r in chunk_records]

    logger.info("Running audio analysis (Fluency/Pronunciation/MTI/Intonation/Engagement)...")
    audio_analysis = analyze_audio(
        audio_path,
        ts_result["full_transcript"],
        ts_result["words"],
        ts_result["sentences"],
        speech_duration,
        total_duration=total_duration,
        speech_chunks=speech_chunks,
    )

    session_id = generate_session_id()
    level1 = build_timeline(
        duration=total_duration,
        sample_rate=sample_rate,
        detected_language=ts_result.get("detected_language"),
        language_probability=ts_result.get("language_probability"),
        speech_chunks=speech_chunks,
        sentences=ts_result["sentences"],
        words=ts_result["words"],
        pronunciation_report=audio_analysis["pronunciation"],
        intonation_report=audio_analysis["intonation"],
        session_id=session_id,
    )
    level2 = build_features(audio_analysis, session_id)

    # Two corrections applied at the boundary rather than inside the analyzers,
    # so the analyzers keep reporting their own raw view and only what leaves
    # this API changes: rhythm_score onto the same 0-100 scale as its
    # neighbours, and metrics the recording is too short to support blanked
    # out instead of published as flattering numbers. See reliability.py.
    reliability.rescale_rhythm(level2)
    reliability_report = reliability.apply(
        level2, speech_seconds=speech_duration, word_count=len(ts_result["words"])
    )
    level2["reliability"] = reliability_report

    return {
        "session_id": session_id,
        "level1": level1,
        "level2": level2,
        "meta": {
            "duration_seconds": round(total_duration, 3),
            "speech_seconds": round(speech_duration, 3),
            "processing_seconds": round(time.time() - start_time, 3),
            "words": len(ts_result["words"]),
            "fully_measured": reliability_report["fully_measured"],
        },
    }


__all__ = ["NoSpeechDetectedError", "extract_features"]
