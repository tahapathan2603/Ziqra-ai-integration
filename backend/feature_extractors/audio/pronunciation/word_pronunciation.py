"""Word-level mispronunciation detection: groups phoneme errors by word and scores severity."""

from typing import Dict, List

from ..phoneme_patterns import known_error_severity, severity_rank
from .phoneme_accuracy import clean_word

_RANK_TO_SEVERITY = {1: "low", 2: "medium", 3: "high"}


def _severity_from_error_count(error_count: int) -> str:
    """1 phoneme error -> low, 2 -> medium, 3+ -> high."""
    if error_count == 1:
        return "low"
    if error_count == 2:
        return "medium"
    return "high"


def _word_severity(word_errors: List[Dict]) -> str:
    """
    Severity for a word, as the higher of two signals:

    1. Error count (more slipped phonemes -> more clearly mispronounced).
    2. The severity MTI assigns to any recognized clarity-affecting pattern
       among the word's errors (e.g. /v/->/w/ is "high" there).

    Taking the max keeps this panel consistent with the MTI panel: a word can't
    read "low" here while MTI flags the same substitution as "high". Words with
    only generic near-misses keep their count-based severity, so ordinary slips
    aren't inflated.
    """
    count_rank = severity_rank(_severity_from_error_count(len(word_errors)))
    pattern_rank = 0
    for err in word_errors:
        sev = known_error_severity(err.get("expected"), err.get("detected"))
        if sev is not None:
            pattern_rank = max(pattern_rank, severity_rank(sev))
    return _RANK_TO_SEVERITY[max(count_rank, pattern_rank)]


def detect_mispronounced_words(phoneme_errors: List[Dict], words: List[Dict]) -> Dict:
    """
    Group phoneme_accuracy.py's per-phoneme errors by word OCCURRENCE and
    attach severity + that occurrence's own timestamps.

    Grouping is by (word text, word_index) — one entry per spoken occurrence,
    not per distinct word type. Grouping by text alone merged every occurrence
    of a repeated word into one entry whose error count was the SUM across all
    of them: measured on a real session, three separate "a"s with one slip each
    became a single "a" with four errors, which the count-based rule below read
    as "high severity mispronunciation" — reported against the first
    occurrence's 20ms span. Per-occurrence grouping reports each slip where it
    actually happened, at its real timestamp, with severity reflecting only
    that occurrence.

    Input:
        phoneme_errors: [{"word": str, "word_index": int, "expected": str,
                          "detected": str}, ...] (the "errors" list from
                         phoneme_accuracy.analyze_phoneme_accuracy()).
                        "word_index" is optional — entries lacking it fall back
                        to first-occurrence timestamp lookup by text.
        words: [{"word": str, "start": float, "end": float}, ...]

    Output: {"mispronounced_words": [{"word": str, "severity": str, "start": float, "end": float}, ...]}
             ordered chronologically by the occurrence's start time.
    """
    errors_by_occurrence: Dict[tuple, List[Dict]] = {}
    for err in phoneme_errors:
        key = (err["word"], err.get("word_index"))
        errors_by_occurrence.setdefault(key, []).append(err)

    # Fallback only, for error entries with no word_index: first-occurrence
    # timestamp per cleaned word.
    first_occurrence: Dict[str, Dict] = {}
    for w in words:
        cleaned = clean_word(w["word"])
        if cleaned not in first_occurrence:
            first_occurrence[cleaned] = {"start": w["start"], "end": w["end"]}

    mispronounced_words = []
    for (word, word_index), word_errors in errors_by_occurrence.items():
        if word_index is not None and 0 <= word_index < len(words):
            spoken = words[word_index]
            ts = {"start": spoken["start"], "end": spoken["end"]}
        else:
            ts = first_occurrence.get(word, {"start": None, "end": None})
        mispronounced_words.append(
            {
                "word": word,
                "severity": _word_severity(word_errors),
                "start": ts["start"],
                "end": ts["end"],
            }
        )

    mispronounced_words.sort(key=lambda w: (w["start"] is None, w["start"]))
    return {"mispronounced_words": mispronounced_words}
