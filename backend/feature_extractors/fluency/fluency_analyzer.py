"""Fluency analysis orchestrator: combines fillers, pauses, and speaking speed."""

from typing import Dict, List

from .fillers import analyze_fillers
from .pauses import analyze_pauses
from .speaking_speed import analyze_speaking_speed


def analyze_fluency(words: List[Dict], sentences: List[Dict], transcript_duration: float) -> Dict:
    """
    Run the full fluency pipeline and combine results into a single report.

    Args:
        words: [{"word": str, "start": float, "end": float}, ...]
        sentences: [{"text": str, "start": float, "end": float}, ...]
        transcript_duration: total speech duration in seconds (used for per-minute rates)

    Returns:
        {
            "fillers": {...},
            "pauses": {...},
            "speaking_speed": {...}
        }
    """
    return {
        "fillers": analyze_fillers(words, transcript_duration),
        "pauses": analyze_pauses(words),
        "speaking_speed": analyze_speaking_speed(words, sentences, transcript_duration),
    }
