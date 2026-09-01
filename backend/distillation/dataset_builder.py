"""
Audio dataset assembly: turns one session's already-computed pipeline outputs
into the two-level dataset record described in the project's audio-dataset
architecture, and writes them to disk.

Level 1 ("timeline") — objective, evidence-only ground truth: audio metadata,
speech chunks, sentence/word timestamps (with real confidence), detected
phonemes (real acoustic timestamps from wav2vec2's CTC output — see
feature_extractors/audio/pronunciation/phoneme_accuracy.py), and raw pitch/
energy contours. Nothing here is an analyzer's interpretation — it does not
change if an analyzer is later replaced or improved, which is the whole point
of keeping it separate from Level 2.

Level 2 ("features") — the interpretation layer: the full output of
feature_extractors/audio/audio_analyzer.py's five analyzers (Fluency,
Pronunciation, MTI, Intonation, Engagement), exactly as they already produce
it (every issue entry is already individually timestamped, so this remains
self-contained without needing to be re-bucketed by chunk — see this
project's dataset-architecture plan for why per-chunk storage was rejected).

Stored as two separate JSON files per session (not one combined file):
Level 1 is stable once generated — VAD/STT/phoneme-extraction never needs to
re-run for a given recording — while Level 2 will legitimately be regenerated
whenever the analyzers themselves improve. Physical separation makes
"re-run only Level 2 for this session" trivial, without ever touching Level 1
again.

TODO(distillation): once the teacher/student loop (train.py, evaluate.py,
incremental_finetune.py) is built, it will read these files as its raw input
for synthetic label generation — no changes needed here for that, this module
only concerns itself with producing a complete, evidence-preserving record.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TIMELINE_DIRNAME = "timeline"
FEATURES_DIRNAME = "features"

# Keys lifted out of the Level 2 intonation copy because they already live in
# Level 1's acoustic_contours (see build_features) — keeping both would
# silently duplicate a several-hundred-point array per session.
_INTONATION_CONTOUR_KEYS = ("pitch_contour", "energy_contour")


def generate_session_id() -> str:
    """Timestamp + short uuid4 hex — human-scannable when browsing dataset/
    directories, and collision-safe across concurrent sessions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{short_uuid}"


def build_timeline(
    duration: float,
    sample_rate: int,
    detected_language: Optional[str],
    language_probability: Optional[float],
    speech_chunks: List[Dict],
    sentences: List[Dict],
    words: List[Dict],
    pronunciation_report: Dict,
    intonation_report: Dict,
    session_id: str,
) -> Dict:
    """
    Assemble Level 1 (timeline / ground truth) for one session.

    Args:
        duration: full recording duration in seconds.
        sample_rate: waveform sample rate in Hz.
        detected_language / language_probability: Whisper's own coarse
            (ISO-639-1) language detection — never a regional/accent variant
            (e.g. never "en-IN"); see speech_to_text.py's docstring for why.
        speech_chunks: VAD-detected speech segments,
            [{"chunk_id": int, "start": float, "end": float}, ...].
        sentences / words: from timestamps.process_timestamps() — words
            include real per-word "confidence" (faster-whisper's own
            probability, not estimated).
        pronunciation_report: analyze_pronunciation() output — used for its
            "detected_phonemes" (real acoustic timestamps).
        intonation_report: analyze_intonation() output — used for its
            "pitch_contour"/"energy_contour" (raw per-frame acoustic ground
            truth).
        session_id: from generate_session_id().

    Returns: the Level 1 timeline dict (see module docstring).
    """
    return {
        "session_id": session_id,
        "audio_metadata": {
            "duration": round(duration, 3),
            "sample_rate": sample_rate,
            "detected_language": detected_language,
            "language_probability": language_probability,
        },
        "speech_chunks": [
            {"chunk_id": c["chunk_id"], "start": c["start"], "end": c["end"]} for c in speech_chunks
        ],
        "sentences": sentences,
        "words": words,
        "detected_phonemes": pronunciation_report.get("detected_phonemes", []),
        "acoustic_contours": {
            "pitch_hz": intonation_report.get("pitch_contour", []),
            "energy_rms": intonation_report.get("energy_contour", []),
        },
    }


def build_features(audio_analysis: Dict, session_id: str) -> Dict:
    """
    Assemble Level 2 (feature extraction / interpretation) for one session.

    Args:
        audio_analysis: feature_extractors.audio.audio_analyzer.analyze_audio()
            output — {"fluency", "pronunciation", "mti", "intonation",
            "engagement", "processing_metadata"}.
        session_id: from generate_session_id() — same value used in the
            timeline for this session, the only linkage key needed between
            the two files (every entry inside is already independently
            timestamped, so no per-item foreign keys are needed either).

    Returns: the Level 2 features dict (see module docstring).
    """
    intonation_copy = {
        k: v for k, v in audio_analysis["intonation"].items() if k not in _INTONATION_CONTOUR_KEYS
    }

    return {
        "session_id": session_id,
        "analysis": {
            "fluency": audio_analysis["fluency"],
            "pronunciation": audio_analysis["pronunciation"],
            "mti": audio_analysis["mti"],
            "intonation": intonation_copy,
            "engagement": audio_analysis["engagement"],
        },
        # Channel quality sits beside the analysis rather than inside it: it
        # describes the recording, not the speaker.
        "audio_quality": audio_analysis.get("audio_quality"),
        "processing_metadata": audio_analysis.get("processing_metadata", {}),
    }


def save_dataset(timeline: Dict, features: Dict, session_id: str, dataset_dir: str) -> Tuple[str, str]:
    """
    Write timeline/features JSON files for one session.

    Returns (timeline_path, features_path).
    """
    timeline_dir = os.path.join(dataset_dir, TIMELINE_DIRNAME)
    features_dir = os.path.join(dataset_dir, FEATURES_DIRNAME)
    os.makedirs(timeline_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    timeline_path = os.path.join(timeline_dir, f"{session_id}.json")
    features_path = os.path.join(features_dir, f"{session_id}.json")

    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset written: {timeline_path}, {features_path}")
    return timeline_path, features_path


def build_and_save_dataset(
    duration: float,
    sample_rate: int,
    detected_language: Optional[str],
    language_probability: Optional[float],
    speech_chunks: List[Dict],
    sentences: List[Dict],
    words: List[Dict],
    audio_analysis: Dict,
    dataset_dir: str,
) -> Dict:
    """
    Convenience wrapper: build both dataset levels for one session and write
    them to disk in one call. This is what callers (e.g. the dev UI) should
    use — build_timeline/build_features/save_dataset are exposed separately
    mainly so Level 2 alone can be rebuilt later without re-deriving Level 1.

    Returns:
        {
            "session_id": str,
            "timeline": {...}, "features": {...},
            "timeline_path": str, "features_path": str,
        }
    """
    session_id = generate_session_id()

    timeline = build_timeline(
        duration=duration,
        sample_rate=sample_rate,
        detected_language=detected_language,
        language_probability=language_probability,
        speech_chunks=speech_chunks,
        sentences=sentences,
        words=words,
        pronunciation_report=audio_analysis["pronunciation"],
        intonation_report=audio_analysis["intonation"],
        session_id=session_id,
    )
    features = build_features(audio_analysis, session_id)
    timeline_path, features_path = save_dataset(timeline, features, session_id, dataset_dir)

    return {
        "session_id": session_id,
        "timeline": timeline,
        "features": features,
        "timeline_path": timeline_path,
        "features_path": features_path,
    }
