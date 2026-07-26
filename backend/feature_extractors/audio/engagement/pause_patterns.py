"""
Pause pattern analysis for engagement: reuses fluency's already-computed pause
data (feature_extractors/fluency/pauses.py) to flag pauses and pause clusters
that specifically hurt engagement — dead air, long pauses, and clusters of
hesitation — rather than every natural pause (fluency already scores those on
their own terms; this module asks a narrower question: does this pause make
the speaker seem disengaged?).

No new audio/timing analysis happens here — reuses fluency's pause list as-is.
"""

from typing import Dict, List

# Heuristic scoring penalties, not independently validated against real
# interview outcomes yet — same spirit as other heuristic scores in this codebase.
DEAD_AIR_PENALTY = 15
LONG_PAUSE_PENALTY = 8
CLUSTER_PENALTY = 10

# Hesitation pauses within this many seconds of each other count as one cluster.
CLUSTER_WINDOW_SECONDS = 5.0
MIN_CLUSTER_SIZE = 2


def _find_hesitation_clusters(hesitations: List[Dict]) -> List[Dict]:
    """Group hesitation-type pauses that occur close together in time."""
    if not hesitations:
        return []

    ordered = sorted(hesitations, key=lambda p: p["start"])
    clusters = []
    current = [ordered[0]]

    for pause in ordered[1:]:
        if pause["start"] - current[-1]["end"] <= CLUSTER_WINDOW_SECONDS:
            current.append(pause)
        else:
            if len(current) >= MIN_CLUSTER_SIZE:
                clusters.append({"start": current[0]["start"], "end": current[-1]["end"], "count": len(current)})
            current = [pause]

    if len(current) >= MIN_CLUSTER_SIZE:
        clusters.append({"start": current[0]["start"], "end": current[-1]["end"], "count": len(current)})

    return clusters


def analyze_pause_patterns(fluency_output: Dict) -> Dict:
    """
    Analyze which pauses specifically hurt engagement (dead air, long pauses,
    hesitation clusters), from fluency's already-computed pause data.

    Input: fluency_output — the full dict returned by
        fluency_analyzer.analyze_fluency() (uses its "pauses" key).

    Output:
        {
            "pause_engagement_score": int,
            "disengaging_pauses": [{"start": float, "end": float, "type": str}, ...],
            "hesitation_clusters": [{"start": float, "end": float, "count": int}, ...],
            "observations": [str, ...],
        }
    """
    all_pauses = fluency_output.get("pauses", {}).get("pauses", [])

    disengaging = [p for p in all_pauses if p["type"] in ("long_pause", "dead_air")]
    hesitations = [p for p in all_pauses if p["type"] == "hesitation"]
    clusters = _find_hesitation_clusters(hesitations)

    disengaging_pauses = [{"start": p["start"], "end": p["end"], "type": p["type"]} for p in disengaging]

    dead_air_count = sum(1 for p in disengaging if p["type"] == "dead_air")
    long_pause_count = sum(1 for p in disengaging if p["type"] == "long_pause")

    penalty = dead_air_count * DEAD_AIR_PENALTY + long_pause_count * LONG_PAUSE_PENALTY + len(clusters) * CLUSTER_PENALTY
    pause_engagement_score = max(0, round(100 - penalty))

    observations = []
    if not disengaging_pauses and not clusters:
        observations.append("No significant disengaging pauses detected.")
    else:
        if dead_air_count:
            observations.append(f"{dead_air_count} dead-air instance(s) significantly disrupt flow.")
        if long_pause_count:
            observations.append(f"{long_pause_count} long pause(s) may reduce perceived engagement.")
        if disengaging_pauses:
            observations.append("Several pauses disrupt the flow of the answer.")
    if clusters:
        observations.append(f"{len(clusters)} hesitation cluster(s) detected, suggesting uncertainty.")

    return {
        "pause_engagement_score": pause_engagement_score,
        "disengaging_pauses": disengaging_pauses,
        "hesitation_clusters": clusters,
        "observations": observations,
    }
