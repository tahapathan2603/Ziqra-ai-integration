"""
Pitch (F0) variation analysis: average/min/max pitch, pitch range, and whether
variation reads as healthy, flat, or erratic.

Uses librosa.yin for the raw pitch track — not pyin, which is ~14x slower in
practice (measured: 12.7s vs 0.9s on ~11.5s of audio) for effectively the same
voiced-frame pitch values, and speed matters here given how long the overall
pipeline already takes. yin doesn't classify voiced/unvoiced itself, and (unlike
pyin) has no built-in confidence measure, so on its own it can return spurious
pitch estimates for noisy/aperiodic frames — measured up to ~2285Hz on real
audio, well outside any human voice. Voicing/reliability is handled here in
three steps: (1) a speech-realistic fmin/fmax range (not librosa's common
music-tutorial default of C2-C7, which extends to ~2093Hz — no one's speaking
voice legitimately reaches that), (2) excluding frames whose estimate sits
right at the fmin/fmax search boundary (a telltale sign yin found no real
periodicity and just returned its search limit), and (3) median-filtering the
contour to suppress isolated octave-jump spikes, a standard, cheap fix for
this well-known pitch-tracker failure mode.
"""

import logging
from typing import Dict, List, Optional

import librosa
import numpy as np
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

FMIN = 60.0   # Hz — below typical low male speaking pitch
FMAX = 500.0  # Hz — above typical high female/child speaking pitch; real speech
              # essentially never goes higher than this in normal delivery
FRAME_LENGTH = 2048
HOP_LENGTH = 512
MEDIAN_FILTER_KERNEL = 5  # frames; smooths isolated octave-jump spikes

# Frames with RMS energy below this percentile of the recording's own RMS
# distribution are treated as silence/unvoiced, not real pitched speech.
VOICED_RMS_PERCENTILE = 40

# Estimates within this fraction of FMIN/FMAX are treated as unreliable
# boundary artifacts, not real pitch.
BOUNDARY_MARGIN = 0.03

# Fix 5 (audio-pipeline correctness-fix plan): compute_pitch_contour's existing
# voicing gate (has_energy & not_boundary) still lets two known librosa.yin
# failure modes through — exact clamp-value returns and isolated octave
# errors — which inflate pitch_range specifically (a single 424.6Hz octave
# spike against a ~64Hz floor read as a 360Hz range for a real ~139Hz
# speaker). clean_pitch() below is the extra pass that catches these.
OCTAVE_REJECT_WINDOW = 15  # frames, for the running-median octave check
OCTAVE_REJECT_HIGH_RATIO = 1.8
OCTAVE_REJECT_LOW_RATIO = 0.55
CLEAN_MEDIAN_FILTER_KERNEL = 5


def _praat_pitch(waveform: np.ndarray, sample_rate: int, times: np.ndarray):
    """
    F0 and voicing from Praat (via parselmouth), sampled onto `times`.

    Praat's autocorrelation pitch tracker decides voiced/unvoiced itself,
    using the same algorithm the phonetics literature is built on, rather
    than the energy-percentile-plus-boundary-margin heuristic that librosa.yin
    needs bolted on (yin returns a number for every frame whether or not
    anything was voiced, which is why that heuristic existed at all).

    The result is resampled onto the caller's existing frame grid deliberately:
    energy_variation.py indexes this function's `voiced` array against its own
    energy frames, so changing the grid would silently misalign the two.

    Returns (f0, voiced) or None if parselmouth is unavailable.
    """
    try:
        import parselmouth
    except ImportError:  # pragma: no cover - dependency is in the image
        return None

    sound = parselmouth.Sound(np.asarray(waveform, dtype=np.float64), sampling_frequency=sample_rate)
    pitch = sound.to_pitch(time_step=HOP_LENGTH / sample_rate, pitch_floor=FMIN, pitch_ceiling=FMAX)
    praat_f0 = pitch.selected_array["frequency"]  # 0.0 where Praat found no voicing
    praat_times = pitch.xs()
    if praat_times.size == 0:
        return None

    # Nearest Praat frame for each frame on our grid.
    idx = np.clip(np.searchsorted(praat_times, times), 0, len(praat_times) - 1)
    left = np.clip(idx - 1, 0, len(praat_times) - 1)
    take_left = np.abs(praat_times[left] - times) < np.abs(praat_times[idx] - times)
    idx = np.where(take_left, left, idx)

    f0 = praat_f0[idx]
    voiced = f0 > 0
    # Unvoiced frames carry 0 Hz from Praat; downstream code reads f0 only
    # where `voiced` is true, but a 0 would poison any accidental mean, so
    # they are filled with the median voiced value instead of a fake pitch.
    if voiced.any():
        f0 = np.where(voiced, f0, float(np.median(f0[voiced])))
    return f0, voiced


def compute_pitch_contour(waveform: np.ndarray, sample_rate: int) -> Dict:
    """
    Compute the pitch contour for a waveform, shared across pitch_variation.py,
    monotonicity.py, and emphasis.py so it's only computed once per recording.

    Praat is the primary tracker (see _praat_pitch); librosa.yin remains as the
    fallback so the pipeline still runs if parselmouth is missing, with the
    energy/boundary voicing heuristic that yin requires.

    Returns: {"f0": np.ndarray, "times": np.ndarray, "voiced": np.ndarray[bool]}
    """
    rms = librosa.feature.rms(y=waveform, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=HOP_LENGTH)

    praat = _praat_pitch(waveform, sample_rate, times)
    if praat is not None:
        f0, voiced = praat
        # Keep the energy gate as a second opinion: Praat can call a very
        # quiet stretch voiced, and a frame with no energy behind it is not
        # something to score delivery on either way.
        energy_threshold = np.percentile(rms, VOICED_RMS_PERCENTILE) if len(rms) else 0.0
        voiced = voiced & (rms > energy_threshold)
        return {"f0": f0, "times": times, "voiced": voiced}

    f0 = librosa.yin(
        waveform, fmin=FMIN, fmax=FMAX, sr=sample_rate,
        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    f0 = medfilt(f0, kernel_size=MEDIAN_FILTER_KERNEL)
    f0 = f0[: len(times)]
    times = times[: len(f0)]
    rms = rms[: len(f0)]

    energy_threshold = np.percentile(rms, VOICED_RMS_PERCENTILE) if len(rms) else 0.0
    has_energy = rms > energy_threshold
    not_boundary = (f0 > FMIN * (1 + BOUNDARY_MARGIN)) & (f0 < FMAX * (1 - BOUNDARY_MARGIN))
    voiced = has_energy & not_boundary

    return {"f0": f0, "times": times, "voiced": voiced}


def _running_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centered running median over `values`, same length as input (edge
    windows shrink rather than pad) — used only to detect octave errors
    against a LOCAL baseline, not to smooth the final signal."""
    half = window // 2
    result = np.empty(len(values))
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        result[i] = np.median(values[lo:hi])
    return result


def clean_pitch(pitch_contour: Dict) -> np.ndarray:
    """
    Return a cleaned array of voiced F0 values for computing pitch STATISTICS
    (average, range, coefficient of variation) — Fix 5. Not for time-indexed
    lookups (order-preserved but not aligned to the contour's own `times`);
    those still use compute_pitch_contour's own voicing gate directly.

    compute_pitch_contour's has_energy/not_boundary gate already rejects a lot
    of noise, but two known librosa.yin failure modes still get through:
    exact clamp-value returns (yin found no periodicity and returned its own
    search boundary — a stricter, exact-value signal than the proportional
    not_boundary margin already applied there) and isolated octave errors
    (real observed case: a 424.6Hz spike and a 64.2Hz halving against a real
    ~139Hz speaker inflated pitch_range to 360.4Hz — over 2x too wide).

    Steps: voiced-only -> drop exact FMIN/FMAX clamp values -> reject octave
    errors (frames far from a local running median are DROPPED, not
    corrected — averaging in a wrong-by-2x value would still bias the result
    even after "fixing" it) -> median-smooth the survivors.
    """
    f0, voiced = pitch_contour["f0"], pitch_contour["voiced"]
    voiced_f0 = f0[voiced]
    if len(voiced_f0) == 0:
        return voiced_f0

    not_clamped = (voiced_f0 != FMIN) & (voiced_f0 != FMAX)
    voiced_f0 = voiced_f0[not_clamped]
    if len(voiced_f0) == 0:
        return voiced_f0

    running_median = _running_median(voiced_f0, OCTAVE_REJECT_WINDOW)
    safe_median = np.where(running_median == 0, 1.0, running_median)
    ratio = voiced_f0 / safe_median
    not_octave_error = (ratio <= OCTAVE_REJECT_HIGH_RATIO) & (ratio >= OCTAVE_REJECT_LOW_RATIO)
    cleaned = voiced_f0[not_octave_error]
    if len(cleaned) < 3:
        return cleaned

    kernel = CLEAN_MEDIAN_FILTER_KERNEL if len(cleaned) >= CLEAN_MEDIAN_FILTER_KERNEL else len(cleaned)
    if kernel % 2 == 0:
        kernel -= 1
    if kernel >= 3:
        cleaned = medfilt(cleaned, kernel_size=kernel)
    return cleaned


def _score_from_cv(cv: float) -> int:
    """
    Heuristic pitch-variation score from the coefficient of variation of voiced
    F0. Natural, expressive speech typically falls roughly in the 0.08-0.25
    range; below that reads as flat/monotone, and this isn't independently
    validated against real interview outcomes — a starting heuristic, same
    spirit as rhythm.py's word-timing CV score.
    """
    if cv < 0.03:
        return 30
    if cv < 0.08:
        return round(30 + (cv - 0.03) / 0.05 * 40)
    if cv < 0.25:
        return round(70 + (cv - 0.08) / 0.17 * 30)
    return 85


def _generate_observations(f0: np.ndarray, times: np.ndarray, voiced: np.ndarray, cv: float) -> List[str]:
    """
    One whole-recording pitch-variation verdict. The exact string
    "Pitch variation is healthy." is load-bearing: intonation_analyzer.py
    replaces it when monotonicity finds sustained flat stretches, so the two
    don't disagree (healthy overall range vs. flat-in-parts). End-of-response
    pitch-trend observations were removed here on purpose — that "fades toward
    the end" narrative is owned by monotonicity.py, so only one module makes it.
    """
    if cv < 0.05:
        return ["Pitch variation is limited; delivery may sound flat."]
    return ["Pitch variation is healthy."]


def analyze_pitch_variation(
    waveform: np.ndarray, sample_rate: int, pitch_contour: Optional[Dict] = None
) -> Dict:
    """
    Analyze pitch variation over a recording.

    Args:
        waveform: mono audio as a numpy array.
        sample_rate: waveform's sample rate in Hz.
        pitch_contour: pre-computed compute_pitch_contour() output, if the
            caller already has it (e.g. intonation_analyzer.py, which shares it
            across pitch/monotonicity/emphasis analysis). Computed here if omitted.

    Returns:
        {
            "pitch_score": int,
            "average_pitch": float,
            "min_pitch": float,
            "max_pitch": float,
            "pitch_range": float,
            "observations": [str, ...],
        }
    """
    if pitch_contour is None:
        pitch_contour = compute_pitch_contour(waveform, sample_rate)
    f0, times, voiced = pitch_contour["f0"], pitch_contour["times"], pitch_contour["voiced"]

    # Fix 5: all statistics below come from the CLEANED voiced F0 (tracker
    # clamp values and octave errors removed), not the raw voiced array —
    # min/max still reported for reference, but pitch_range is p95-p5, never
    # raw max-min, since a single octave-error survivor would otherwise still
    # blow out the range even after cleaning removes most such frames.
    cleaned_f0 = clean_pitch(pitch_contour)
    if len(cleaned_f0) == 0:
        return {
            "pitch_score": 0,
            "average_pitch": 0.0,
            "min_pitch": 0.0,
            "max_pitch": 0.0,
            "pitch_range": 0.0,
            "observations": ["No voiced speech detected to analyze pitch."],
        }

    average_pitch = float(np.mean(cleaned_f0))
    min_pitch = float(np.min(cleaned_f0))
    max_pitch = float(np.max(cleaned_f0))
    pitch_range = float(np.percentile(cleaned_f0, 95) - np.percentile(cleaned_f0, 5))
    cv = float(np.std(cleaned_f0) / average_pitch) if average_pitch else 0.0

    return {
        "pitch_score": _score_from_cv(cv),
        "average_pitch": round(average_pitch, 1),
        "min_pitch": round(min_pitch, 1),
        "max_pitch": round(max_pitch, 1),
        "pitch_range": round(pitch_range, 1),
        "observations": _generate_observations(f0, times, voiced, cv),
    }
