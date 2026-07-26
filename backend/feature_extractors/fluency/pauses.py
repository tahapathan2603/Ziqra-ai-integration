"""Inter-word pause detection and classification."""

from typing import Dict, List


def classify_pause(duration: float) -> str:
    """< 0.5s -> natural, 0.5-1.5s -> hesitation, 1.5-3s -> long_pause, > 3s -> dead_air."""
    if duration < 0.5:
        return "natural"
    if duration < 1.5:
        return "hesitation"
    if duration <= 3.0:
        return "long_pause"
    return "dead_air"


def detect_pauses(words: List[Dict]) -> List[Dict]:
    """
    Detect gaps between consecutive words.

    pause_duration = next_word.start - current_word.end

    Input: [{"word": str, "start": float, "end": float}, ...] (chronological order)
    Output: [{"start": float, "end": float, "duration": float, "type": str}, ...]
    """
    pauses = []
    for current, nxt in zip(words, words[1:]):
        duration = nxt["start"] - current["end"]
        if duration > 0:
            pauses.append(
                {
                    "start": current["end"],
                    "end": nxt["start"],
                    "duration": duration,
                    "type": classify_pause(duration),
                }
            )
    return pauses


def analyze_pauses(words: List[Dict]) -> Dict:
    """
    Run the full pause analysis pipeline.

    Returns:
        {
            "total_pauses": int,
            "pauses": [{"start": float, "end": float, "duration": float, "type": str}, ...]
        }
    """
    pauses = detect_pauses(words)
    return {
        "total_pauses": len(pauses),
        "pauses": pauses,
    }
