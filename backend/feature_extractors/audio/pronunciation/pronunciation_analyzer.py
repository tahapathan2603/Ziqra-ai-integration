"""Pronunciation analysis orchestrator: combines phoneme accuracy, word-level
mispronunciation, stress placement, and rhythm into a single report."""

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

import torchaudio

from .phoneme_accuracy import analyze_phoneme_accuracy
from .rhythm import analyze_rhythm
from .stress_placement import analyze_stress
from .word_pronunciation import detect_mispronounced_words

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _load_waveform(audio_path: str, sample_rate: int = SAMPLE_RATE):
    """
    Load audio as a mono waveform at `sample_rate` (matches what the phoneme
    recognizer expects). Normalizes via ffmpeg first, same approach as
    preprocessing/silero_vad.py's load_audio — duplicated here as a small,
    self-contained helper rather than importing across packages, since silero_vad's
    loader has its own sys.path handling tied to being run as a script directly.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ac", "1",
            "-ar", str(sample_rate),
            "-vn",
            "-loglevel", "error",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode '{audio_path}': {result.stderr.strip()}")
        waveform, sr = torchaudio.load(tmp_path)
        return waveform.squeeze(0), sr
    finally:
        os.remove(tmp_path)


def _overall_label(score: int) -> str:
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Needs Improvement"
    return "Weak"


def _generate_overall_observations(
    pronunciation_score: int,
    mispronounced_words: List[Dict],
    stress_errors: List[Dict],
    rhythm_issues: List[str],
) -> List[str]:
    observations = [f"Overall pronunciation is {_overall_label(pronunciation_score).lower()}."]

    if mispronounced_words:
        high_severity = [w for w in mispronounced_words if w["severity"] == "high"]
        if high_severity:
            # Fix 7: the count must always match what's actually named. Real
            # observed bug: "6 word(s)... : thank, samad, periwal, study,
            # basketball." (5 names, count 6) — the count came from the full
            # list, the names from a silently truncated slice of it. Truncate
            # the DISPLAY only, and say so explicitly when truncated, rather
            # than quoting a total that doesn't match what's listed.
            shown = high_severity[:5]
            names = ", ".join(w["word"] for w in shown)
            if len(high_severity) > len(shown):
                observations.append(f"{len(high_severity)} word(s) show significant mispronunciation, including: {names}.")
            else:
                observations.append(f"{len(high_severity)} word(s) show significant mispronunciation: {names}.")
        else:
            observations.append(f"{len(mispronounced_words)} word(s) show minor pronunciation deviations.")
    else:
        observations.append("No significant mispronunciations detected.")

    if stress_errors:
        observations.append(f"{len(stress_errors)} word(s) show incorrect stress placement.")

    if rhythm_issues:
        observations.append("Speech rhythm shows some irregularity (" + ", ".join(rhythm_issues) + ").")

    return observations


def analyze_pronunciation(
    audio_path: Optional[str],
    transcript: str,
    words: List[Dict],
    sentences: Optional[List[Dict]] = None,
    speech_chunks: Optional[List[Dict]] = None,
) -> Dict:
    """
    Run the full pronunciation pipeline and combine results into one report.

    Exposes every sub-analyzer's output (not just a trimmed summary) so the full
    pipeline can be inspected — phoneme-level errors, per-word mispronunciations
    with timestamps, stress errors, and rhythm issues are all included, not just
    the four top-line scores.

    Args:
        audio_path: path to the interview audio — loaded here and passed to
            phoneme_accuracy.py for real acoustic phoneme detection.
        transcript: full transcript text (kept for interface completeness; word-level
            data is read from `words`, so this isn't used directly yet).
        words: [{"word": str, "start": float, "end": float}, ...]
        sentences: [{"text": str, "start": float, "end": float}, ...] (from
            timestamps.process_timestamps()) — phoneme detection runs per
            sentence, not per word (see phoneme_accuracy.py for why). Falls
            back to treating all of `words` as one sentence if omitted.
        speech_chunks: VAD speech regions (Level 1 ground truth) — Fix 8:
            restricts rhythm's onset-interval calculation to word pairs
            inside the same chunk. Optional; falls back to no gating.

    Returns:
        {
            "pronunciation_score": int,
            "phoneme_accuracy": float,
            "stress_accuracy": float,
            "rhythm_score": float,
            "phoneme_errors": [{"word": str, "expected": str, "detected": str}, ...],
            "detected_phonemes": [{"phoneme": str, "start": float, "end": float}, ...],
            "mispronounced_words": [{"word": str, "severity": str, "start": float, "end": float}, ...],
            "stress_errors": [{"word": str, "expected": str, "detected": str}, ...],
            "rhythm_issues": [str, ...],
            "skipped_words": [{"word": str, "reason": str}, ...],
                # Words never scored (proper names, low-confidence/truncated
                # fragments — see exclusions.py). Reported for transparency,
                # never counted as errors or folded into phoneme_accuracy.
            "clip_edge_errors": [{"word": str, "expected": str}, ...],
                # Phoneme positions the decoder never covered (recording/chunk
                # boundary truncation) — excluded from scoring, kept here for
                # audit visibility. See phoneme_accuracy.py's clip-edge detection.
            "experimental_fields": [str, ...],
                # Fix 8: which top-level fields above are documented starting
                # heuristics, not independently validated — currently
                # "stress_accuracy"/"stress_errors" (hard placeholder) and
                # "rhythm_score"/"rhythm_issues" (real signal, uncalibrated
                # thresholds). A coach-model training-dataset builder should
                # treat these as directional, not ground truth.
            "overall_observations": [str, ...],
        }
    """
    logger.info("Loading audio for phoneme analysis...")
    waveform, sample_rate = _load_waveform(audio_path)

    logger.info("Analyzing phoneme accuracy...")
    phoneme_report = analyze_phoneme_accuracy(words, sentences or [], waveform, sample_rate)

    logger.info("Detecting mispronounced words...")
    mispronounced_report = detect_mispronounced_words(phoneme_report["errors"], words)

    logger.info("Analyzing stress placement...")
    stress_report = analyze_stress(words)

    logger.info("Analyzing rhythm...")
    rhythm_report = analyze_rhythm(words, speech_chunks=speech_chunks)

    # stress_accuracy is intentionally NOT part of this composite: stress
    # detection is still a placeholder that always reports 100% (see
    # stress_placement.py), so averaging it in would inflate every score toward
    # 100 regardless of real delivery. Add stress back into the average once
    # real acoustic stress detection lands.
    #
    # Phoneme accuracy carries the weight, rhythm contributes a quarter. They
    # used to be averaged 50/50, which let a heuristic that rhythm.py's own
    # docstring calls "directional, not authoritative" move the headline
    # pronunciation number by up to 50 points. It is real signal — Fix 8 made
    # it discriminate between sessions — but it measures inter-word timing,
    # not pronunciation, and it is not calibrated against any outcome. A
    # quarter keeps it visible in the composite without letting it outvote the
    # measurement the score is named after.
    pronunciation_score = round(
        100 * (0.75 * phoneme_report["phoneme_accuracy"] + 0.25 * rhythm_report["rhythm_score"])
    )
    logger.info(f"Pronunciation analysis complete. Score: {pronunciation_score}")

    return {
        "pronunciation_score": pronunciation_score,
        "phoneme_accuracy": phoneme_report["phoneme_accuracy"],
        # How strongly the acoustics supported each expected phoneme, from the
        # same CTC pass the accuracy above comes from (see gop.py). Reported,
        # not yet scored: turning it into a calibrated number needs labelled
        # pronunciation data, and it already earns its place by removing
        # "errors" the audio actually supports.
        "gop": phoneme_report.get("gop"),
        # Underscore-prefixed: consumed by scoring.py inside this process and
        # dropped before publication (see audio_analyzer).
        "_gop_scores": phoneme_report.get("_gop_scores", []),
        "_expected_phonemes": phoneme_report.get("_expected_phonemes", 0),
        "stress_accuracy": stress_report["stress_accuracy"],
        "rhythm_score": rhythm_report["rhythm_score"],
        "phoneme_errors": phoneme_report["errors"],
        "detected_phonemes": phoneme_report["detected_phonemes"],
        "mispronounced_words": mispronounced_report["mispronounced_words"],
        "stress_errors": stress_report["stress_errors"],
        "rhythm_issues": rhythm_report["issues"],
        "skipped_words": phoneme_report["skipped_words"],
        "clip_edge_errors": phoneme_report["clip_edge_errors"],
        "experimental_fields": ["stress_accuracy", "stress_errors", "rhythm_score", "rhythm_issues"],
        "overall_observations": _generate_overall_observations(
            pronunciation_score,
            mispronounced_report["mispronounced_words"],
            stress_report["stress_errors"],
            rhythm_report["issues"],
        ),
    }
