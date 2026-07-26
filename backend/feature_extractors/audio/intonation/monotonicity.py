"""
Monotonicity analysis: sustained stretches of speech with little pitch
variation ("flat" delivery), as opposed to pitch_variation.py's single
whole-recording pitch-range summary.

Reuses the same pitch contour as pitch_variation.py (no separate pitch
extraction) via a sliding window over the F0 track, looking for sustained
low-variation windows rather than just an overall range.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pitch_variation import compute_pitch_contour

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 3.0
STEP_SECONDS = 1.0
MIN_VOICED_FRAMES_PER_WINDOW = 3

# A window's voiced-pitch coefficient of variation below this reads as flat.
MONOTONE_CV_THRESHOLD = 0.04

# Merged flat windows shorter than this aren't reported as their own section —
# brief flat stretches (e.g. a short pause-adjacent word) are normal.
MIN_MONOTONE_SECTION_SECONDS = 3.0

# A candidate window must be at least this much inside a real VAD speech chunk
# to be considered at all (Fix 4) — otherwise a window straddling a silence
# gap could gather enough scattered voiced frames from its edges to look flat.
MIN_VOICED_RATIO = 0.5


def _speech_mask(times: np.ndarray, speech_chunks) -> np.ndarray:
    """Boolean mask, aligned to `times`, marking frames inside any VAD speech
    chunk. All-True if speech_chunks is omitted (no gating). See Fix 4 in the
    audio-pipeline correctness-fix plan — same rationale as
    energy_variation.py's identical helper (duplicated rather than shared,
    matching this package's existing per-module small-helper convention)."""
    if not speech_chunks:
        return np.ones(len(times), dtype=bool)
    mask = np.zeros(len(times), dtype=bool)
    for chunk in speech_chunks:
        mask |= (times >= chunk["start"]) & (times <= chunk["end"])
    return mask


def _windowed_cv(
    f0: np.ndarray, times: np.ndarray, voiced: np.ndarray, speech_mask: np.ndarray
) -> List[Tuple[float, float, float]]:
    """
    Sliding-window coefficient of variation of voiced F0. Returns (start, end,
    cv) tuples. A window is only considered if MOST of its frames are inside a
    real speech chunk (Fix 4) — otherwise a window straddling a silence gap
    could still gather enough scattered voiced frames from its edges to look
    artificially flat.
    """
    if len(times) == 0:
        return []

    windows = []
    total = float(times[-1])
    t = 0.0
    while t < total:
        window_end = min(t + WINDOW_SECONDS, total)
        in_window = (times >= t) & (times < window_end)
        if in_window.any() and float(np.mean(speech_mask[in_window])) < MIN_VOICED_RATIO:
            t += STEP_SECONDS
            continue
        mask = voiced & in_window
        voiced_f0 = f0[mask]
        if len(voiced_f0) >= MIN_VOICED_FRAMES_PER_WINDOW:
            mean = np.mean(voiced_f0)
            cv = float(np.std(voiced_f0) / mean) if mean else 0.0
            windows.append((t, window_end, cv))
        t += STEP_SECONDS

    return windows


def _merge_flat_windows(flagged: List[Tuple[float, float, float]]) -> List[Dict]:
    """Merge overlapping/adjacent flagged windows into reportable sections."""
    if not flagged:
        return []

    sections = []
    cur_start, cur_end, cvs = flagged[0][0], flagged[0][1], [flagged[0][2]]
    for start, end, cv in flagged[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            cvs.append(cv)
        else:
            sections.append((cur_start, cur_end, cvs))
            cur_start, cur_end, cvs = start, end, [cv]
    sections.append((cur_start, cur_end, cvs))

    result = []
    for start, end, cvs in sections:
        if end - start >= MIN_MONOTONE_SECTION_SECONDS:
            avg_cv = sum(cvs) / len(cvs)
            severity = "high" if avg_cv < MONOTONE_CV_THRESHOLD * 0.5 else "medium"
            result.append({"start": round(start, 2), "end": round(end, 2), "severity": severity})
    return result


def analyze_monotonicity(
    waveform: np.ndarray,
    sample_rate: int,
    pitch_contour: Optional[Dict] = None,
    speech_chunks: Optional[List[Dict]] = None,
) -> Dict:
    """
    Detect sustained monotone (flat pitch) sections in a recording.

    Args:
        waveform: mono audio as a numpy array.
        sample_rate: waveform's sample rate in Hz.
        pitch_contour: pre-computed compute_pitch_contour() output (from
            pitch_variation.py), if the caller already has it. Computed here
            if omitted.
        speech_chunks: VAD speech regions (Level 1 ground truth) — Fix 4:
            windows outside real speech are never considered flat. Falls back
            to no gating if omitted.

    Returns:
        {
            "monotonicity_score": int,   # higher = more natural variation, less monotone
            "monotone_sections": [{"start": float, "end": float, "severity": str}, ...],
            "observations": [str, ...],
        }
    """
    if pitch_contour is None:
        pitch_contour = compute_pitch_contour(waveform, sample_rate)
    f0, times, voiced = pitch_contour["f0"], pitch_contour["times"], pitch_contour["voiced"]

    speech_mask = _speech_mask(times, speech_chunks)
    windows = _windowed_cv(f0, times, voiced, speech_mask)
    flagged = [w for w in windows if w[2] < MONOTONE_CV_THRESHOLD]
    monotone_sections = _merge_flat_windows(flagged)

    total_duration = float(times[-1]) if len(times) else 0.0
    monotone_time = sum(s["end"] - s["start"] for s in monotone_sections)
    fraction_monotone = (monotone_time / total_duration) if total_duration else 0.0
    monotonicity_score = max(0, round(100 * (1 - fraction_monotone * 1.3)))

    observations = []
    if not monotone_sections:
        observations.append("No sustained monotone sections detected.")
    else:
        observations.append(f"Detected {len(monotone_sections)} monotone section(s).")
        if total_duration and monotone_sections[-1]["end"] >= total_duration - 5:
            observations.append("Speech becomes flat near the conclusion.")

    return {
        "monotonicity_score": monotonicity_score,
        "monotone_sections": monotone_sections,
        "observations": observations,
    }
