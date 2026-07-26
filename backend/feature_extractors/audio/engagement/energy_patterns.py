"""
Energy pattern analysis for engagement: reframes intonation's already-computed
energy signal (real RMS analysis, see feature_extractors/audio/intonation/
energy_variation.py) through an engagement lens.

No new audio analysis happens here — per the engagement module's philosophy,
this intelligently reuses upstream signals instead of recomputing them.
"""

from typing import Dict, List


def _generate_observations(low_energy_segments: List[Dict]) -> List[str]:
    """
    One engagement-lens line about energy. Deliberately does NOT echo
    intonation's own energy observations (segment lists, second-half trend) —
    those already appear in the Intonation section, and repeating them here made
    the report say the same thing twice. This states only the engagement takeaway.
    """
    if not low_energy_segments:
        return ["Energy stays consistent, reading as steady enthusiasm."]

    high_severity = [s for s in low_energy_segments if s["severity"] == "high"]
    if high_severity:
        return [f"{len(high_severity)} noticeable energy dip(s) may read as reduced enthusiasm."]
    return ["Minor energy dips, but enthusiasm largely holds."]


def analyze_energy_patterns(intonation_output: Dict) -> Dict:
    """
    Analyze energy consistency and enthusiasm patterns from intonation's energy data.

    Input: intonation_output — the full dict returned by
        intonation_analyzer.analyze_intonation() (uses its "energy_variation" key).

    Output:
        {
            "energy_engagement_score": int,
            "low_energy_segments": [{"start": float, "end": float, "severity": str}, ...],
            "observations": [str, ...],
        }
    """
    energy_variation = intonation_output.get("energy_variation", {})
    energy_score = energy_variation.get("energy_score", 0)
    low_energy_segments = energy_variation.get("low_energy_sections", [])

    return {
        "energy_engagement_score": energy_score,
        "low_energy_segments": low_energy_segments,
        "observations": _generate_observations(low_energy_segments),
    }
