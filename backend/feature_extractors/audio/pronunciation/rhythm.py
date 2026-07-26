"""
Speech rhythm analysis, based on inter-word onset-interval timing variability.

Natural speech has real variation in pacing; very uniform, near-identical
timing reads as mechanical/robotic delivery, while highly irregular timing
(long unpredictable gaps mixed with rapid bursts) reads as halting/disfluent.
(This is a *timing* signal — vocal flatness/monotone pitch is intonation's
concern, kept separate on purpose so the two don't report the same word
twice.) This is a lightweight timing-based heuristic, not full acoustic
prosody analysis, and — see Fix 8's "experimental" flag below — not yet
independently validated against real interview outcomes; treat rhythm_score
as directional, not authoritative, until it is.

Fix 8 (audio-pipeline correctness-fix plan): the previous version measured
only each word's OWN duration (end - start), which never reflects the SILENCE
between words at all — a 1.9s mid-utterance gap was invisible to it. Natural
word-duration variability alone (driven mostly by word length, not pacing
quality) also happened to fall inside the old scoring band on every real
session tested, so it always returned a flat 1.0/no-issues regardless of
actual pacing — a second placeholder alongside stress, not a real measurement.

Fixed by using ONSET-INTERVAL timing instead (word[i+1].start - word[i].start,
which captures a word's duration AND its trailing gap together — the actual
pacing signal), restricted to consecutive word pairs inside the SAME VAD
speech chunk (Fix 4's ownership rule: silence BETWEEN chunks belongs to
Fluency's pause classifier, not rhythm). Verified against the three real QA
sessions: chunk-gated onset-interval CV was 0.537/0.558 for two clean
sessions and 0.631 for the session with a real 1.9s disfluent gap — a
detectable, genuine separation the old word-duration CV (0.43/0.54/0.58,
all inside the same dead zone) never produced.
"""

import statistics
from typing import Dict, List, Optional

# Center and tolerance of the onset-interval CV band real natural speech
# lands in — informed by the three real QA sessions (observed 0.537-0.631),
# NOT independently validated against labeled interview-quality outcomes.
# Scores fall off smoothly (not a flat plateau) the further CV sits from this
# center in EITHER direction, so the metric always has room to discriminate
# rather than saturating at a fixed value across a wide dead zone.
IDEAL_CV_CENTER = 0.55
MIN_SCORE = 0.3

# Below this, timing reads as suspiciously uniform/mechanical.
MONOTONE_CV_THRESHOLD = 0.25
# Above this, timing reads as erratic/halting.
ERRATIC_CV_THRESHOLD = 0.85


def _coefficient_of_variation(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / mean


def _same_speech_chunk(a: Dict, b: Dict, speech_chunks: List[Dict]) -> bool:
    for chunk in speech_chunks:
        if chunk["start"] <= a["start"] <= chunk["end"] and chunk["start"] <= b["start"] <= chunk["end"]:
            return True
    return False


def _onset_intervals(words: List[Dict], speech_chunks: Optional[List[Dict]]) -> List[float]:
    """Time between consecutive words' START timestamps — captures a word's
    own duration plus its trailing gap in one measure. Pairs that straddle a
    speech-chunk boundary are skipped (that silence is Fluency's territory,
    not rhythm's — same ownership rule as Fix 4)."""
    intervals = []
    for a, b in zip(words, words[1:]):
        if speech_chunks is not None and not _same_speech_chunk(a, b, speech_chunks):
            continue
        intervals.append(b["start"] - a["start"])
    return intervals


def _score_from_cv(cv: float) -> float:
    """Continuous falloff from IDEAL_CV_CENTER — no flat plateau, so nearby
    but distinct CVs still produce distinct scores (the concrete discrimination
    problem this fix addresses)."""
    if cv <= 0:
        return MIN_SCORE
    relative_distance = abs(cv - IDEAL_CV_CENTER) / IDEAL_CV_CENTER
    return max(MIN_SCORE, 1.0 - relative_distance)


def analyze_rhythm(words: List[Dict], speech_chunks: Optional[List[Dict]] = None) -> Dict:
    """
    Score speech rhythm from inter-word onset-interval timing variability.

    Args:
        words: [{"word": str, "start": float, "end": float}, ...]
        speech_chunks: VAD speech regions ([{"start": float, "end": float}, ...],
            Level 1 ground truth) — restricts onset intervals to pairs inside
            the same chunk. Falls back to no gating if omitted.

    Output: {"rhythm_score": float, "issues": [str, ...], "experimental": True}
        "experimental" (Fix 8): this metric is a documented starting
        heuristic, not independently validated against real interview
        outcomes — downstream consumers (e.g. a coach-model training-dataset
        builder) should treat it as directional, not ground truth, same as
        stress_placement.py's placeholder.
    """
    if len(words) < 3:
        return {"rhythm_score": 1.0, "issues": [], "experimental": True}

    intervals = _onset_intervals(words, speech_chunks)
    if len(intervals) < 2:
        return {"rhythm_score": 1.0, "issues": [], "experimental": True}

    cv = _coefficient_of_variation(intervals)

    issues = []
    if cv < MONOTONE_CV_THRESHOLD:
        issues.append("uniform word timing")
    elif cv > ERRATIC_CV_THRESHOLD:
        issues.append("erratic pacing")

    score = _score_from_cv(cv)
    return {"rhythm_score": round(min(1.0, max(0.0, score)), 2), "issues": issues, "experimental": True}
