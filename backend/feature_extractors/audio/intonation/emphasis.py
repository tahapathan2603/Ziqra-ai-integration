"""
Emphasis analysis: whether important (content) words get vocal emphasis — a
pitch or energy peak relative to the recording's own baseline — versus being
delivered flatly like function words.

Word importance uses a stopword-based heuristic (anything not in a common
function-word list, and longer than 2 letters, is treated as a content-word
candidate) rather than a real POS tagger — this avoids adding an NLP model
dependency for what's meant to be a lightweight signal. A real POS tagger
(e.g. nltk's averaged_perceptron_tagger, or spaCy) would identify action
verbs/key nouns more precisely; swap is_content_word() for that when available.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from ..phoneme_patterns import FUNCTION_WORDS
from ..pronunciation.phoneme_accuracy import clean_word
from .energy_variation import compute_energy_contour
from .pitch_variation import compute_pitch_contour

logger = logging.getLogger(__name__)

# A word's peak energy/pitch during its span must exceed the recording's own
# median by these ratios to count as "emphasized". Heuristic thresholds, not
# independently validated against real interview outcomes.
ENERGY_EMPHASIS_RATIO = 1.15
PITCH_EMPHASIS_RATIO = 1.08


def is_content_word(word: str) -> bool:
    cleaned = clean_word(word)
    return bool(cleaned) and len(cleaned) > 2 and cleaned not in FUNCTION_WORDS


def _peak_in_span(values: np.ndarray, times: np.ndarray, start: float, end: float) -> Optional[float]:
    mask = (times >= start) & (times <= end)
    span_values = values[mask]
    return float(np.max(span_values)) if len(span_values) else None


def analyze_emphasis(
    words: List[Dict],
    waveform: np.ndarray,
    sample_rate: int,
    pitch_contour: Optional[Dict] = None,
    energy_contour: Optional[Dict] = None,
) -> Dict:
    """
    Detect content words that lack vocal emphasis.

    Args:
        words: [{"word": str, "start": float, "end": float}, ...]
        waveform: mono audio as a numpy array.
        sample_rate: waveform's sample rate in Hz.
        pitch_contour / energy_contour: pre-computed contours, if the caller
            already has them. Computed here if omitted.

    Returns:
        {
            "emphasis_score": int,
            "under_emphasized_words": [{"word": str, "timestamp": {"start": float, "end": float}}, ...],
            "observations": [str, ...],
        }
    """
    if pitch_contour is None:
        pitch_contour = compute_pitch_contour(waveform, sample_rate)
    if energy_contour is None:
        energy_contour = compute_energy_contour(waveform, sample_rate)

    f0, f0_times, voiced = pitch_contour["f0"], pitch_contour["times"], pitch_contour["voiced"]
    rms, rms_times = energy_contour["rms"], energy_contour["times"]

    voiced_f0 = f0[voiced]
    baseline_pitch = float(np.median(voiced_f0)) if len(voiced_f0) else 0.0
    baseline_energy = float(np.median(rms)) if len(rms) else 0.0

    important_words = [w for w in words if is_content_word(w["word"])]

    under_emphasized = []
    for w in important_words:
        peak_energy = _peak_in_span(rms, rms_times, w["start"], w["end"])
        peak_pitch = _peak_in_span(f0[voiced], f0_times[voiced], w["start"], w["end"])

        energy_emphasized = (
            peak_energy is not None and baseline_energy > 0 and peak_energy > baseline_energy * ENERGY_EMPHASIS_RATIO
        )
        pitch_emphasized = (
            peak_pitch is not None and baseline_pitch > 0 and peak_pitch > baseline_pitch * PITCH_EMPHASIS_RATIO
        )

        if not (energy_emphasized or pitch_emphasized):
            under_emphasized.append(
                {
                    "word": clean_word(w["word"]),
                    "timestamp": {"start": w["start"], "end": w["end"]},
                }
            )

    emphasis_score = (
        round(100 * (1 - len(under_emphasized) / len(important_words))) if important_words else 100
    )

    observations = []
    if not important_words:
        observations.append("No content words identified to assess emphasis.")
    elif not under_emphasized:
        observations.append("Important words receive adequate vocal emphasis.")
    else:
        observations.append(
            f"{len(under_emphasized)} of {len(important_words)} important word(s) lack clear emphasis."
        )
        if len(under_emphasized) / len(important_words) > 0.5:
            observations.append(
                "Delivery may sound flat on key content words; consider stressing action verbs and key nouns."
            )

    return {
        "emphasis_score": emphasis_score,
        "under_emphasized_words": under_emphasized,
        "observations": observations,
    }
