"""Filler word/phrase detection and scoring."""

import re
from typing import Dict, List

# Add new fillers here — single words or multi-word phrases both work.
# Longer phrases are matched greedily before shorter ones (see _FILLER_PATTERNS).
FILLERS = [
    "um",
    "uh",
    "like",
    "you know",
    "basically",
    "actually",
    "kind of",
    "sort of",
]

_FILLER_PATTERNS = sorted(
    (tuple(phrase.lower().split()) for phrase in FILLERS),
    key=len,
    reverse=True,
)


def _normalize(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def detect_fillers(words: List[Dict]) -> List[Dict]:
    """
    Scan a word list for filler words/phrases (case-insensitive, punctuation-stripped).

    Input: [{"word": str, "start": float, "end": float}, ...]
    Output: [{"word": str, "start": float, "end": float}, ...] — one entry per filler
    occurrence, "word" holding the matched filler text (may span multiple words).
    """
    normalized = [_normalize(w["word"]) for w in words]
    fillers = []
    i = 0
    n = len(words)

    while i < n:
        matched = False
        for pattern in _FILLER_PATTERNS:
            plen = len(pattern)
            if i + plen <= n and tuple(normalized[i : i + plen]) == pattern:
                fillers.append(
                    {
                        "word": " ".join(pattern),
                        "start": words[i]["start"],
                        "end": words[i + plen - 1]["end"],
                    }
                )
                i += plen
                matched = True
                break
        if not matched:
            i += 1

    return fillers


def calculate_fillers_per_minute(fillers: List[Dict], duration: float) -> float:
    """Fillers per minute, given total speech duration in seconds."""
    if duration <= 0:
        return 0.0
    return len(fillers) / (duration / 60.0)


def classify_filler_usage(rate: float) -> str:
    """0-2/min -> good, 3-5/min -> moderate, 6+/min -> needs_improvement."""
    if rate <= 2:
        return "good"
    if rate <= 5:
        return "moderate"
    return "needs_improvement"


def analyze_fillers(words: List[Dict], duration: float) -> Dict:
    """
    Run the full filler analysis pipeline.

    Returns:
        {
            "filler_count": int,
            "fillers_per_minute": float,
            "classification": str,
            "fillers": [{"word": str, "start": float, "end": float}, ...]
        }
    """
    fillers = detect_fillers(words)
    rate = calculate_fillers_per_minute(fillers, duration)
    return {
        "filler_count": len(fillers),
        "fillers_per_minute": round(rate, 1),
        "classification": classify_filler_usage(rate),
        "fillers": fillers,
    }
