"""
Intonation analysis orchestrator: combines pitch variation, energy variation,
monotonicity, and emphasis into a single report.

Goal is not just scores — every dimension here returns observations and (where
applicable) timestamped problem sections, so the report can explain *why* a
score is what it is, not just state a number. See individual files for how each
score is computed; none of these thresholds are independently validated against
real interview outcomes yet — they're documented starting heuristics.

Pitch/energy contours are computed once here and shared across all four
sub-analyzers (pitch_variation.py's compute_pitch_contour() is also reused
by monotonicity.py and emphasis.py; energy_variation.py's compute_energy_contour()
is also reused by emphasis.py) rather than re-extracted per analyzer.
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np
import torchaudio

from .emphasis import analyze_emphasis
from .energy_variation import analyze_energy_variation, compute_energy_contour
from .monotonicity import analyze_monotonicity
from .pitch_variation import analyze_pitch_variation, compute_pitch_contour

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _load_waveform(audio_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio as a mono numpy waveform at `sample_rate`. Same self-contained
    ffmpeg-normalize approach as pronunciation_analyzer.py / mti_analyzer.py's
    loaders, adapted to return numpy (what librosa expects) instead of a torch
    tensor.
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
        waveform, _sr = torchaudio.load(tmp_path)
        return waveform.squeeze(0).numpy()
    finally:
        os.remove(tmp_path)


# Delivery-specific vocabulary, deliberately distinct from the engagement
# module's "Highly Engaging / Moderately Engaging" scale. Intonation grades how
# expressive vs. flat the *voice* is; engagement grades the interviewer's
# overall impression. Using different words stops the two sections from
# appearing to contradict each other over the same recording.
def _serialize_pitch_contour(pitch_contour: Dict) -> List[Dict]:
    """
    Convert the raw per-frame pitch contour (numpy arrays, internal-only until
    now) into a plain JSON-serializable list. This is real acoustic ground
    truth already computed here for pitch_variation.py/monotonicity.py/
    emphasis.py — exposing it costs nothing extra (no recomputation), and lets
    a dataset consumer see the actual pitch curve shape, not just summary
    stats derived from it.
    """
    f0, times, voiced = pitch_contour["f0"], pitch_contour["times"], pitch_contour["voiced"]
    return [
        {"time": round(float(t), 3), "hz": round(float(f), 1), "voiced": bool(v)}
        for t, f, v in zip(times, f0, voiced)
    ]


def _serialize_energy_contour(energy_contour: Dict) -> List[Dict]:
    """Same as _serialize_pitch_contour, for the raw RMS energy contour."""
    rms, times = energy_contour["rms"], energy_contour["times"]
    return [{"time": round(float(t), 3), "rms": round(float(r), 5)} for t, r in zip(times, rms)]


def _overall_delivery_label(score: int) -> str:
    if score >= 85:
        return "Expressive"
    if score >= 70:
        return "Moderately expressive"
    if score >= 50:
        return "Somewhat flat"
    return "Flat"


# Below this monotonicity score, the recording has sustained flat stretches, so
# a whole-file "pitch variation is healthy" verdict would be misleading.
_MONOTONE_RECONCILE_THRESHOLD = 60


def _reconcile_pitch_with_monotonicity(pitch_report: Dict, monotonicity_report: Dict) -> None:
    """
    pitch_variation.py measures the whole recording's pitch range; monotonicity.py
    looks for sustained flat *windows*. A recording can have a healthy overall
    range while still going flat for long stretches — reported naively that reads
    as a contradiction ("Pitch variation is healthy." next to "Detected 2
    monotone sections."). When monotonicity is low, rewrite the healthy-pitch
    line in place so both point the same way. Mutates pitch_report.
    """
    if monotonicity_report["monotonicity_score"] >= _MONOTONE_RECONCILE_THRESHOLD:
        return
    reconciled = "Pitch varies overall, but delivery stays flat for sustained stretches."
    pitch_report["observations"] = [
        reconciled if obs == "Pitch variation is healthy." else obs
        for obs in pitch_report["observations"]
    ]


def _combine_observations(
    score: int,
    pitch_report: Dict,
    energy_report: Dict,
    monotonicity_report: Dict,
    emphasis_report: Dict,
) -> List[str]:
    observations = [f"Overall delivery: {_overall_delivery_label(score)}."]

    if energy_report["low_energy_sections"]:
        observations.append("Energy drops in parts of the response.")
    if monotonicity_report["monotone_sections"]:
        observations.append("Some sections of speech lack vocal variation.")
    if emphasis_report["under_emphasized_words"]:
        observations.append("Several important words lack emphasis.")

    if not (
        energy_report["low_energy_sections"]
        or monotonicity_report["monotone_sections"]
        or emphasis_report["under_emphasized_words"]
    ):
        observations.append("No major delivery issues detected.")

    return observations


def analyze_intonation(
    audio_path: Optional[str],
    transcript: str,
    words: List[Dict],
    speech_chunks: Optional[List[Dict]] = None,
) -> Dict:
    """
    Run the full intonation pipeline and combine results into one report.

    Args:
        audio_path: path to the interview audio.
        transcript: full transcript text (kept for interface completeness;
            emphasis.py works from `words`, not raw transcript text).
        words: [{"word": str, "start": float, "end": float}, ...]
        speech_chunks: VAD speech regions ([{"start": float, "end": float}, ...],
            Level 1 ground truth) — Fix 4 (audio-pipeline correctness-fix
            plan): threaded into energy_variation/monotonicity so silence
            (leading, or between chunks) is never scored as a delivery
            problem — that's exclusively Fluency's pause classifier's concern.
            Optional; falls back to unfiltered behavior if omitted.

    Returns:
        {
            "intonation_score": int,
            "delivery_label": str,   # Expressive | Moderately expressive | Somewhat flat | Flat
            "pitch_variation": {...},
            "energy_variation": {...},
            "monotonicity": {...},
            "emphasis": {...},
            "overall_observations": [str, ...],
            "pitch_contour": [{"time": float, "hz": float, "voiced": bool}, ...],
            "energy_contour": [{"time": float, "rms": float}, ...],
                # Raw per-frame acoustic ground truth (~every 32ms), kept as
                # separate top-level keys rather than nested in pitch_variation/
                # energy_variation so a dataset builder can lift them out
                # without touching those sub-dicts' internals.
        }
    """
    logger.info("Loading audio for intonation analysis...")
    waveform = _load_waveform(audio_path)

    logger.info("Computing pitch contour...")
    pitch_contour = compute_pitch_contour(waveform, SAMPLE_RATE)
    logger.info("Computing energy contour...")
    energy_contour = compute_energy_contour(waveform, SAMPLE_RATE)

    logger.info("Analyzing pitch variation...")
    pitch_report = analyze_pitch_variation(waveform, SAMPLE_RATE, pitch_contour=pitch_contour)

    logger.info("Analyzing energy variation...")
    energy_report = analyze_energy_variation(
        waveform, SAMPLE_RATE, energy_contour=energy_contour,
        speech_chunks=speech_chunks, voiced=pitch_contour["voiced"],
    )

    logger.info("Analyzing monotonicity...")
    monotonicity_report = analyze_monotonicity(
        waveform, SAMPLE_RATE, pitch_contour=pitch_contour, speech_chunks=speech_chunks,
    )

    logger.info("Analyzing emphasis...")
    emphasis_report = analyze_emphasis(
        words, waveform, SAMPLE_RATE, pitch_contour=pitch_contour, energy_contour=energy_contour
    )

    # Reconcile the two pitch-flatness views before anything is reported, so the
    # whole-file pitch verdict and the windowed monotonicity finding agree.
    _reconcile_pitch_with_monotonicity(pitch_report, monotonicity_report)

    intonation_score = round(
        np.mean(
            [
                pitch_report["pitch_score"],
                energy_report["energy_score"],
                monotonicity_report["monotonicity_score"],
                emphasis_report["emphasis_score"],
            ]
        )
    )

    overall_observations = _combine_observations(
        intonation_score, pitch_report, energy_report, monotonicity_report, emphasis_report
    )

    logger.info(f"Intonation analysis complete. Score: {intonation_score}")

    return {
        "intonation_score": intonation_score,
        "delivery_label": _overall_delivery_label(intonation_score),
        "pitch_variation": pitch_report,
        "energy_variation": energy_report,
        "monotonicity": monotonicity_report,
        "emphasis": emphasis_report,
        "overall_observations": overall_observations,
        "pitch_contour": _serialize_pitch_contour(pitch_contour),
        "energy_contour": _serialize_energy_contour(energy_contour),
        # Raw arrays for the fitted scorers (scoring.py), so they don't have to
        # recompute a pitch track that was just computed here. Stripped before
        # anything is published — see dataset_builder._INTONATION_CONTOUR_KEYS.
        "_arrays": {
            "f0": pitch_contour["f0"],
            "voiced": pitch_contour["voiced"],
            "rms": energy_contour["rms"],
        },
    }
