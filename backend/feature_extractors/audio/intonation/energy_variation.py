"""
Energy (loudness) variation analysis: RMS energy over time, low-energy sections,
and whether energy is sustained or drops off.
"""

import logging
from typing import Dict, List, Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

FRAME_LENGTH = 2048
HOP_LENGTH = 512

# "Low energy" is defined relative to the recording's own *speech* loudness:
# frames quieter than this fraction of the median speech-frame RMS. Relative
# (not an absolute dB threshold) because raw amplitude varies a lot by mic/setup.
# Using a fraction-of-median rather than a fixed percentile matters: a percentile
# threshold flags a fixed share of every recording (e.g. the old 25th percentile
# always called ~25% of frames "low", even in a perfectly steady, loud answer),
# which manufactured low-energy sections that then contradicted a high energy
# score. A ratio flags nothing when energy really is consistent.
LOW_ENERGY_RATIO = 0.5

# Frames below this percentile of RMS are treated as non-speech (silence/breath)
# and excluded from the median that defines the "speech loudness" baseline, so
# leading/trailing silence doesn't drag the baseline down.
SPEECH_FRAME_PERCENTILE = 40

# Minimum sustained duration for a low-energy stretch to be worth reporting as
# its own section, rather than just a brief natural dip (e.g. a breath).
MIN_LOW_ENERGY_SECTION_SECONDS = 1.0

# A candidate low-energy section must be at least this voiced (fraction of its
# frames marked voiced in the pitch contour) to be reported — otherwise a
# breathy/silent trailing stretch inside an otherwise-fine speech chunk can
# still fire the detector even after chunk-gating below.
MIN_VOICED_RATIO = 0.5


def _speech_mask(times: np.ndarray, speech_chunks: Optional[List[Dict]]) -> np.ndarray:
    """
    Boolean mask, aligned to `times`, marking frames that fall inside any VAD
    speech chunk (Level 1 ground truth — see distillation/dataset_builder.py).
    Returns an all-True mask if speech_chunks is omitted (caller didn't have
    it), so this degrades to "no gating" rather than erroring.

    Fix 4 (audio-pipeline correctness-fix plan): silence between/around real
    speech was being scored as "low energy delivery" — leading silence before
    the speaker starts, and inter-chunk pauses already counted by Fluency's
    pause classifier. Ownership: silence between speech chunks belongs
    EXCLUSIVELY to Fluency's pause classifier; this module only ever looks
    inside chunks. Do not remove this gating without re-introducing that
    double-count.
    """
    if not speech_chunks:
        return np.ones(len(times), dtype=bool)
    mask = np.zeros(len(times), dtype=bool)
    for chunk in speech_chunks:
        mask |= (times >= chunk["start"]) & (times <= chunk["end"])
    return mask


def compute_energy_contour(waveform: np.ndarray, sample_rate: int) -> Dict:
    """
    Compute the RMS energy contour for a waveform, shared across
    energy_variation.py and emphasis.py so it's only computed once per recording.

    Returns: {"rms": np.ndarray, "times": np.ndarray}
    """
    rms = librosa.feature.rms(y=waveform, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=HOP_LENGTH)
    return {"rms": rms, "times": times}


def _find_low_energy_sections(
    rms: np.ndarray,
    times: np.ndarray,
    threshold: float,
    speech_mask: np.ndarray,
    voiced: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Merge contiguous below-threshold frames into sections of at least
    MIN_LOW_ENERGY_SECTION_SECONDS — restricted to frames inside `speech_mask`
    (Fix 4): a masked-out (non-speech) frame resets any in-progress run, the
    same as an above-threshold frame would, so silence never starts, extends,
    or bridges a low-energy section. If `voiced` is given, a candidate section
    is only kept when at least MIN_VOICED_RATIO of its frames are voiced —
    guards against a breathy/silent stretch inside an otherwise-real chunk.
    """
    sections = []
    start_idx = None

    def _close_section(end_idx):
        start_time, end_time = float(times[start_idx]), float(times[end_idx])
        if end_time - start_time < MIN_LOW_ENERGY_SECTION_SECONDS:
            return
        if voiced is not None:
            voiced_ratio = float(np.mean(voiced[start_idx : end_idx + 1]))
            if voiced_ratio < MIN_VOICED_RATIO:
                return
        avg = float(np.mean(rms[start_idx : end_idx + 1]))
        severity = "high" if avg < threshold * 0.5 else "medium"
        sections.append({"start": round(start_time, 2), "end": round(end_time, 2), "severity": severity})

    for i, (val, in_speech) in enumerate(zip(rms, speech_mask)):
        if in_speech and val < threshold:
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            _close_section(i - 1)
            start_idx = None
    if start_idx is not None:
        _close_section(len(rms) - 1)

    return sections


def _generate_observations(rms: np.ndarray, times: np.ndarray, low_energy_sections: List[Dict]) -> List[str]:
    """
    One energy verdict, no self-contradiction. Reports a directional trend
    (energy fades / builds over the answer) OR the count of low-energy sections
    OR "consistent" — never "consistent" alongside "decreases", which the old
    two-independent-checks version could produce.
    """
    first_avg = second_avg = None
    if len(times):
        midpoint = times[-1] / 2
        first_half = rms[times < midpoint]
        second_half = rms[times >= midpoint]
        if len(first_half) and len(second_half):
            first_avg, second_avg = float(np.mean(first_half)), float(np.mean(second_half))

    if first_avg is not None and second_avg < first_avg * 0.85:
        return ["Energy fades over the course of the answer, which can read as running out of steam."]
    if first_avg is not None and second_avg > first_avg * 1.15:
        return ["Energy builds over the course of the answer."]
    if low_energy_sections:
        return [f"Detected {len(low_energy_sections)} low-energy section(s), but overall energy holds up."]
    return ["Energy remains consistent throughout the response."]


def _score_energy(low_energy_sections: List[Dict], total_duration: float) -> int:
    if total_duration <= 0:
        return 0
    low_energy_time = sum(s["end"] - s["start"] for s in low_energy_sections)
    fraction_low = low_energy_time / total_duration
    return max(0, round(100 * (1 - fraction_low * 1.5)))


def analyze_energy_variation(
    waveform: np.ndarray,
    sample_rate: int,
    energy_contour: Optional[Dict] = None,
    speech_chunks: Optional[List[Dict]] = None,
    voiced: Optional[np.ndarray] = None,
) -> Dict:
    """
    Analyze loudness variation over a recording.

    Args:
        waveform: mono audio as a numpy array.
        sample_rate: waveform's sample rate in Hz.
        energy_contour: pre-computed compute_energy_contour() output, if the
            caller already has it. Computed here if omitted.
        speech_chunks: VAD speech regions ([{"start": float, "end": float}, ...],
            Level 1 ground truth) — Fix 4: restricts both the speech-loudness
            baseline and low-energy-section search to inside these regions, so
            silence (leading, or between chunks — already Fluency's pause
            classifier's concern) is never scored as a delivery problem.
            Falls back to no gating if omitted (matches prior behavior).
        voiced: pitch contour's voiced mask, aligned to this contour's own
            `times` (both use the same FRAME_LENGTH/HOP_LENGTH) — if given,
            requires a candidate low-energy section be mostly voiced.

    Returns:
        {
            "energy_score": int,
            "low_energy_sections": [{"start": float, "end": float, "severity": str}, ...],
            "observations": [str, ...],
        }
    """
    if energy_contour is None:
        energy_contour = compute_energy_contour(waveform, sample_rate)
    rms, times = energy_contour["rms"], energy_contour["times"]

    if len(rms) == 0:
        return {"energy_score": 0, "low_energy_sections": [], "observations": ["No audio to analyze."]}

    speech_mask = _speech_mask(times, speech_chunks)

    # Baseline = median RMS of speech frames only (exclude the quietest
    # frames, which are silence/breaths) — restricted to in-chunk frames so
    # out-of-chunk silence can never drag the baseline down either.
    in_chunk_rms = rms[speech_mask] if speech_mask.any() else rms
    speech_floor = float(np.percentile(in_chunk_rms, SPEECH_FRAME_PERCENTILE))
    speech_frames = in_chunk_rms[in_chunk_rms >= speech_floor]
    speech_baseline = float(np.median(speech_frames)) if len(speech_frames) else float(np.median(in_chunk_rms))
    threshold = speech_baseline * LOW_ENERGY_RATIO
    low_energy_sections = _find_low_energy_sections(rms, times, threshold, speech_mask, voiced=voiced)
    total_duration = float(times[-1]) if len(times) else 0.0

    return {
        "energy_score": _score_energy(low_energy_sections, total_duration),
        "low_energy_sections": low_energy_sections,
        "observations": _generate_observations(rms, times, low_energy_sections),
    }
