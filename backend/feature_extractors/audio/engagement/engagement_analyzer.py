"""
Engagement analysis orchestrator: combines fluency, pronunciation, MTI, and
intonation signals into a single report answering "would an interviewer
perceive this speaker as energetic, interested, natural, and confident?"

Philosophy: this module recomputes nothing from scratch. Every score here is
built from data already produced by the other four modules — no new audio
analysis, no new model calls, just synthesis and higher-level reasoning over
existing signals. That also makes this module very fast (pure Python, no
audio/model work) compared to the rest of the pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .energy_patterns import analyze_energy_patterns
from .pause_patterns import analyze_pause_patterns
from .speaking_dynamics import analyze_speaking_dynamics

logger = logging.getLogger(__name__)

TIMELINE_WINDOW_SECONDS = 30.0

# Fraction of a timeline window covered by "problem" sections (low-energy,
# monotone, disengaging pauses) above which that window is rated low/medium
# engagement. Heuristic, not independently validated against real interview
# outcomes yet.
LOW_ENGAGEMENT_THRESHOLD = 0.4
MEDIUM_ENGAGEMENT_THRESHOLD = 0.15

# Top-level engagement_score weighting across the three sub-analyzers plus a
# smaller contribution from MTI's clarity score — clarity issues can undermine
# perceived confidence/engagement even when delivery itself is lively.
SCORE_WEIGHTS = {"energy": 0.30, "pause": 0.25, "dynamics": 0.35, "clarity": 0.10}

# Max points subtracted from the blended score when problem sections cover the
# whole recording. This couples the headline score to the timeline: a recording
# whose windows are mostly "low engagement" can't also post a top score. Without
# it, the weighted average could read "highly engaging" while every timeline
# window said "low" (the exact contradiction this coupling removes).
COVERAGE_PENALTY_MAX = 30.0


def _estimate_total_duration(
    fluency_output: Dict, pronunciation_output: Dict, mti_output: Dict, intonation_output: Dict
) -> float:
    """
    Estimate the recording's total duration from the latest timestamp found
    anywhere across the four upstream reports.

    analyze_engagement() only receives those four reports, not the audio or
    word list directly, so there's no exact duration available — this is a
    reasonable approximation (real engagement analysis usually has *something*
    timestamped near the end of a recording), but it will underestimate for a
    recording with literally zero flagged issues anywhere. Pass total_duration
    explicitly (the caller usually has it, e.g. from VAD) to avoid relying on
    this estimate.
    """
    ends = [0.0]

    for p in fluency_output.get("pauses", {}).get("pauses", []):
        ends.append(p.get("end") or 0.0)
    for f in fluency_output.get("fillers", {}).get("fillers", []):
        ends.append(f.get("end") or 0.0)

    for w in pronunciation_output.get("mispronounced_words", []):
        ends.append(w.get("end") or 0.0)

    for key in ("vowel_patterns", "consonant_patterns", "stress_transfer"):
        for item in mti_output.get(key, []):
            ends.append(item.get("timestamp", {}).get("end") or 0.0)

    for s in intonation_output.get("energy_variation", {}).get("low_energy_sections", []):
        ends.append(s.get("end") or 0.0)
    for s in intonation_output.get("monotonicity", {}).get("monotone_sections", []):
        ends.append(s.get("end") or 0.0)
    for w in intonation_output.get("emphasis", {}).get("under_emphasized_words", []):
        ends.append(w.get("timestamp", {}).get("end") or 0.0)

    return max(ends)


def _collect_problem_sections(energy_report: Dict, pause_report: Dict, intonation_output: Dict) -> List[Dict]:
    sections = []
    for s in energy_report["low_energy_segments"]:
        sections.append({"start": s["start"], "end": s["end"]})
    for p in pause_report["disengaging_pauses"]:
        sections.append({"start": p["start"], "end": p["end"]})
    for c in pause_report["hesitation_clusters"]:
        sections.append({"start": c["start"], "end": c["end"]})
    for s in intonation_output.get("monotonicity", {}).get("monotone_sections", []):
        sections.append({"start": s["start"], "end": s["end"]})
    return sections


def _merged_coverage(problem_sections: List[Dict], total_duration: float) -> float:
    """
    Fraction of the recording (0.0-1.0) covered by problem sections, with
    overlapping sections merged so the same second isn't counted twice (e.g. a
    stretch that's both low-energy and monotone). Drives the score's coverage
    penalty so it lines up with the timeline, which buckets the same sections.
    """
    if total_duration <= 0 or not problem_sections:
        return 0.0

    intervals = sorted(((s["start"], s["end"]) for s in problem_sections), key=lambda x: x[0])
    merged_total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged_total += cur_end - cur_start
            cur_start, cur_end = start, end
    merged_total += cur_end - cur_start
    return min(1.0, merged_total / total_duration)


_LEVEL_ORDER = ["low_engagement", "needs_improvement", "moderately_engaging", "highly_engaging"]


def _cap_level_by_timeline(level: str, timeline: List[Dict]) -> str:
    """
    Stop the headline level from outrunning the timeline. If most windows read
    "low", the level can't exceed "needs_improvement"; if any window reads "low",
    it can't be "highly_engaging". Guarantees the label and the per-window
    verdict never flatly contradict each other.
    """
    if not timeline:
        return level

    low_windows = sum(1 for w in timeline if w["engagement"] == "low")
    idx = _LEVEL_ORDER.index(level)
    if low_windows > len(timeline) / 2:
        idx = min(idx, _LEVEL_ORDER.index("needs_improvement"))
    elif low_windows > 0:
        idx = min(idx, _LEVEL_ORDER.index("moderately_engaging"))
    return _LEVEL_ORDER[idx]


def _build_timeline(total_duration: float, problem_sections: List[Dict]) -> List[Dict]:
    """Bucket the recording into fixed windows and rate each by how much of it overlaps problem sections."""
    if total_duration <= 0:
        return []

    timeline = []
    t = 0.0
    while t < total_duration:
        window_end = min(t + TIMELINE_WINDOW_SECONDS, total_duration)
        window_duration = window_end - t

        overlap = sum(max(0.0, min(window_end, s["end"]) - max(t, s["start"])) for s in problem_sections)
        fraction_bad = (overlap / window_duration) if window_duration > 0 else 0.0

        if fraction_bad > LOW_ENGAGEMENT_THRESHOLD:
            level = "low"
        elif fraction_bad > MEDIUM_ENGAGEMENT_THRESHOLD:
            level = "medium"
        else:
            level = "high"

        timeline.append({"start": round(t), "end": round(window_end), "engagement": level})
        t += TIMELINE_WINDOW_SECONDS

    return timeline


def _classify_engagement_level(score: int) -> str:
    if score >= 85:
        return "highly_engaging"
    if score >= 70:
        return "moderately_engaging"
    if score >= 50:
        return "needs_improvement"
    return "low_engagement"


def _collect_strengths_and_improvements(
    energy_report: Dict, pause_report: Dict, dynamics_report: Dict, mti_output: Dict, pronunciation_output: Dict
) -> Tuple[List[str], List[str]]:
    strengths = list(dynamics_report["strengths"])
    improvement_areas = list(dynamics_report["issues"])

    if energy_report["energy_engagement_score"] >= 80 and not energy_report["low_energy_segments"]:
        strengths.append("Consistent energy throughout the response")
    for s in energy_report["low_energy_segments"]:
        if s["severity"] == "high":
            improvement_areas.append(f"Significant energy drop between {s['start']:.1f}s and {s['end']:.1f}s")

    if pause_report["pause_engagement_score"] >= 80 and not pause_report["disengaging_pauses"]:
        strengths.append("Smooth flow with no disruptive pauses")
    elif pause_report["disengaging_pauses"]:
        improvement_areas.append("Long pauses reduce engagement")

    # Pronunciation's high-severity mispronunciations and MTI's clarity-risk
    # words are two views of the same underlying problem (a word that's hard to
    # understand) and overlap heavily now that both grade severity from the same
    # phoneme tables. Merge them into one deduped line instead of two that name
    # the same words.
    clarity_words = [
        w["word"] for w in pronunciation_output.get("mispronounced_words", []) if w.get("severity") == "high"
    ]
    clarity_words += mti_output.get("clarity_risk_words", [])
    clarity_words = list(dict.fromkeys(clarity_words))
    if clarity_words:
        improvement_areas.append("Clarity issues on key word(s): " + ", ".join(clarity_words[:5]))

    # Dedupe while preserving order.
    strengths = list(dict.fromkeys(strengths))
    improvement_areas = list(dict.fromkeys(improvement_areas))
    return strengths, improvement_areas


def _check_engagement_invariants(
    improvement_areas: List[str], fluency_output: Dict, mti_output: Dict
) -> None:
    """
    Fix 7: Engagement must never claim a clean bill of delivery health while an
    upstream module reported a real problem (real observed damage: engagement
    97/"highly_engaging" while Fluency reported speaking_speed "too_fast").
    Logs a warning rather than raising — a coherence bug here shouldn't crash
    the pipeline, but it must be visible, not silently shipped. Reused
    verbatim by the Fix 10 regression fixtures as the same check against real
    session data, so this stays the one place the invariant is defined.
    """
    speed_classification = fluency_output.get("speaking_speed", {}).get("classification")
    non_ideal_pace = speed_classification not in (None, "ideal")
    high_clarity_impact = mti_output.get("clarity_impact") == "high"

    if (non_ideal_pace or high_clarity_impact) and not improvement_areas:
        logger.warning(
            "Engagement invariant violated: reported zero improvement_areas despite "
            f"speaking_speed={speed_classification!r}, mti_clarity_impact={mti_output.get('clarity_impact')!r}."
        )


def analyze_engagement(
    fluency_output: Dict,
    pronunciation_output: Dict,
    mti_output: Dict,
    intonation_output: Dict,
    total_duration: Optional[float] = None,
) -> Dict:
    """
    Run the full engagement pipeline and combine results into one report.

    Args:
        fluency_output: fluency_analyzer.analyze_fluency() output.
        pronunciation_output: pronunciation_analyzer.analyze_pronunciation() output.
        mti_output: mti_analyzer.analyze_mti() output.
        intonation_output: intonation_analyzer.analyze_intonation() output.
        total_duration: recording duration in seconds, if the caller has it
            (e.g. from VAD) — used to build the timeline. If omitted, it's
            estimated from the latest timestamp found across all four reports
            (see _estimate_total_duration).

    Returns:
        {
            "engagement_score": int,
            "engagement_level": str,
            "strengths": [str, ...],
            "improvement_areas": [str, ...],
            "timeline": [{"start": int, "end": int, "engagement": str}, ...],
            "energy_patterns": {...},
            "pause_patterns": {...},
            "speaking_dynamics": {...},
        }
    """
    logger.info("Analyzing energy patterns...")
    energy_report = analyze_energy_patterns(intonation_output)

    logger.info("Analyzing pause patterns...")
    pause_report = analyze_pause_patterns(fluency_output)

    logger.info("Analyzing speaking dynamics...")
    dynamics_report = analyze_speaking_dynamics(fluency_output, pronunciation_output, intonation_output)

    if total_duration is None:
        total_duration = _estimate_total_duration(
            fluency_output, pronunciation_output, mti_output, intonation_output
        )
    problem_sections = _collect_problem_sections(energy_report, pause_report, intonation_output)
    timeline = _build_timeline(total_duration, problem_sections)

    clarity_score = mti_output.get("overall_mti_score", 100)
    base_score = (
        SCORE_WEIGHTS["energy"] * energy_report["energy_engagement_score"]
        + SCORE_WEIGHTS["pause"] * pause_report["pause_engagement_score"]
        + SCORE_WEIGHTS["dynamics"] * dynamics_report["dynamics_score"]
        + SCORE_WEIGHTS["clarity"] * clarity_score
    )
    # Couple the score to how much of the recording is actually problematic, so
    # the headline number moves with the timeline instead of against it.
    coverage_penalty = COVERAGE_PENALTY_MAX * _merged_coverage(problem_sections, total_duration)
    engagement_score = max(0, min(100, round(base_score - coverage_penalty)))

    # Derive the level from the (penalized) score, then cap it so it can never
    # claim more than the timeline supports.
    engagement_level = _cap_level_by_timeline(_classify_engagement_level(engagement_score), timeline)

    strengths, improvement_areas = _collect_strengths_and_improvements(
        energy_report, pause_report, dynamics_report, mti_output, pronunciation_output
    )
    _check_engagement_invariants(improvement_areas, fluency_output, mti_output)

    logger.info(f"Engagement analysis complete. Score: {engagement_score} ({engagement_level})")

    return {
        "engagement_score": engagement_score,
        "engagement_level": engagement_level,
        "strengths": strengths,
        "improvement_areas": improvement_areas,
        "timeline": timeline,
        "energy_patterns": energy_report,
        "pause_patterns": pause_report,
        "speaking_dynamics": dynamics_report,
    }
