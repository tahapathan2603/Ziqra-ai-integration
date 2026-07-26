"""
Evidence packet: the single, LLM-facing serialization of all five audio
analysis reports (Fluency, Pronunciation, MTI, Intonation, Engagement).

This is intentionally the ONLY input the LLM feedback layer sees — narrower
than the full reports, so the LLM narrates decisions the feature extractors
already made rather than re-deriving anything (see feedback_generator.py's
grounding rules in its system prompt). Two things are deliberately excluded:

- Raw contours / per-word timestamp dumps: expensive in tokens, not useful
  for narrative coaching feedback.
- All stress fields (stress_accuracy / stress_errors / stress_transfer):
  stress_placement.py's _detect_stress() is a placeholder that always returns
  the expected pattern (see that file's own PLACEHOLDER notice), so
  stress_accuracy always reads 100% and stress_errors is always empty.
  Feeding that to an LLM would produce confident, false praise about
  "excellent stress placement" for something never actually measured.
"""

from typing import Dict, List, Optional

from backend.feature_extractors.audio.phoneme_patterns import severity_rank

MAX_ITEMS_PER_LIST = 10


def _cap_by_severity(items: List[Dict]) -> List[Dict]:
    """Keep the top MAX_ITEMS_PER_LIST items, highest severity first."""
    if not items:
        return []
    ranked = sorted(items, key=lambda i: severity_rank(i.get("severity", "")), reverse=True)
    return ranked[:MAX_ITEMS_PER_LIST]


def _cap(items: List, limit: int = MAX_ITEMS_PER_LIST) -> List:
    """No severity field to rank by (e.g. fillers, timeline windows) — keep
    chronological order and just truncate."""
    return items[:limit] if items else []


def _round_ts(value: Optional[float]) -> Optional[float]:
    return round(value, 2) if isinstance(value, (int, float)) else value


def _mispronounced_words(pronunciation_report: Dict) -> List[Dict]:
    words = _cap_by_severity(pronunciation_report.get("mispronounced_words", []))
    return [
        {
            "word": w["word"],
            "severity": w["severity"],
            "start": _round_ts(w.get("start")),
            "end": _round_ts(w.get("end")),
        }
        for w in words
    ]


def _mti_patterns(mti_report: Dict) -> List[Dict]:
    combined = list(mti_report.get("vowel_patterns", [])) + list(mti_report.get("consonant_patterns", []))
    combined = _cap_by_severity(combined)
    return [
        {
            "word": p["word"],
            "issue": p["issue"],
            "expected_phoneme": p["expected"],
            "detected_phoneme": p["detected"],
            "severity": p["severity"],
            "start": _round_ts(p.get("timestamp", {}).get("start")),
            "end": _round_ts(p.get("timestamp", {}).get("end")),
        }
        for p in combined
    ]


def _flat_sections(intonation_report: Dict) -> List[Dict]:
    monotone = [
        {**s, "kind": "monotone"} for s in intonation_report.get("monotonicity", {}).get("monotone_sections", [])
    ]
    low_energy = [
        {**s, "kind": "low_energy"}
        for s in intonation_report.get("energy_variation", {}).get("low_energy_sections", [])
    ]
    combined = _cap_by_severity(monotone + low_energy)
    return [
        {
            "kind": s["kind"],
            "start": _round_ts(s.get("start")),
            "end": _round_ts(s.get("end")),
            "severity": s.get("severity"),
        }
        for s in combined
    ]


def build_evidence_packet(
    fluency_report: Dict,
    pronunciation_report: Dict,
    mti_report: Dict,
    intonation_report: Dict,
    engagement_report: Dict,
    transcript: str,
    total_duration: float,
) -> Dict:
    """
    Build the compact, structured packet the LLM feedback layer reasons over.

    Args: the five analyzer output dicts (see each analyzer's own docstring
        for full shape), plus the transcript text and recording duration.

    Returns: a nested dict — see the module docstring for what's kept/dropped.
        Every list is capped at MAX_ITEMS_PER_LIST entries, highest-severity
        first where severity exists.
    """
    fillers = fluency_report.get("fillers", {})
    pauses = fluency_report.get("pauses", {}).get("pauses", [])
    speaking_speed = fluency_report.get("speaking_speed", {})

    return {
        "transcript": transcript,
        "duration_seconds": _round_ts(total_duration),
        "fluency": {
            "speaking_speed": {
                "words_per_minute": speaking_speed.get("words_per_minute"),
                "classification": speaking_speed.get("classification"),
            },
            "filler_usage": {
                "count": fillers.get("filler_count", 0),
                "per_minute": fillers.get("fillers_per_minute", 0.0),
                "classification": fillers.get("classification"),
                "examples": _cap([f["word"] for f in fillers.get("fillers", [])]),
            },
            "pauses": {
                "total": len(pauses),
                "dead_air_count": sum(1 for p in pauses if p.get("type") == "dead_air"),
                "long_pause_count": sum(1 for p in pauses if p.get("type") == "long_pause"),
                "hesitation_count": sum(1 for p in pauses if p.get("type") == "hesitation"),
            },
        },
        "pronunciation": {
            "score": pronunciation_report.get("pronunciation_score"),
            "phoneme_accuracy_pct": round(pronunciation_report.get("phoneme_accuracy", 0) * 100),
            "rhythm_score_pct": round(pronunciation_report.get("rhythm_score", 0) * 100),
            "observations": pronunciation_report.get("overall_observations", []),
            "mispronounced_words": _mispronounced_words(pronunciation_report),
        },
        "mti": {
            "score": mti_report.get("overall_mti_score"),
            "clarity_impact": mti_report.get("clarity_impact"),
            "top_recurring_issue": mti_report.get("top_recurring_issue"),
            "observations": mti_report.get("overall_observations", []),
            "clarity_risk_words": _cap(mti_report.get("clarity_risk_words", [])),
            "patterns": _mti_patterns(mti_report),
        },
        "intonation": {
            "score": intonation_report.get("intonation_score"),
            "delivery_label": intonation_report.get("delivery_label"),
            "observations": intonation_report.get("overall_observations", []),
            "pitch": {
                "average_hz": intonation_report.get("pitch_variation", {}).get("average_pitch"),
                "range_hz": intonation_report.get("pitch_variation", {}).get("pitch_range"),
            },
            "flat_sections": _flat_sections(intonation_report),
        },
        "engagement": {
            "score": engagement_report.get("engagement_score"),
            "level": engagement_report.get("engagement_level"),
            "strengths": _cap(engagement_report.get("strengths", [])),
            "improvement_areas": _cap(engagement_report.get("improvement_areas", [])),
            "timeline": _cap(engagement_report.get("timeline", [])),
        },
    }
