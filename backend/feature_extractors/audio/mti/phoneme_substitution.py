"""
Shared word-timestamp helper for the MTI modules.

NOTE: this file previously also flagged known phoneme substitutions
(e.g. /θ/ -> /t/). That role was removed because every substitution it knew
about is a consonant-to-consonant shift already classified by
consonant_patterns.py — running both double-counted the same error (two panels,
two penalties, sometimes two different severities). Consonant substitution
classification now lives solely in consonant_patterns.py, with its tables in
feature_extractors/audio/phoneme_patterns.py. Only build_word_timestamps()
remains here, imported by the other mti/ modules.
"""

from typing import Dict, List

from ..pronunciation.phoneme_accuracy import clean_word


def build_word_timestamps(words: List[Dict]) -> Dict[str, Dict]:
    """
    First-occurrence start/end timestamp per cleaned word.

    Shared across all mti/ modules (imported from here, the same pattern
    pronunciation/phoneme_accuracy.py's clean_word() follows for its siblings).

    Only a FALLBACK now — prefer resolve_error_timestamp() below, which pins an
    error to the specific occurrence it actually came from.
    """
    timestamps: Dict[str, Dict] = {}
    for w in words:
        cleaned = clean_word(w["word"])
        if cleaned not in timestamps:
            timestamps[cleaned] = {"start": w["start"], "end": w["end"]}
    return timestamps


def resolve_error_timestamp(err: Dict, words: List[Dict], fallback: Dict[str, Dict]) -> Dict:
    """
    Start/end timestamp for the specific word OCCURRENCE a phoneme error came
    from, using the error's "word_index" (index into `words`) when present.

    Falls back to first-occurrence-by-text (`fallback`, from
    build_word_timestamps) for error entries produced before word_index
    existed. Without the index, a repeated word's every error is stamped with
    its first occurrence's timing, which points the user at the wrong moment in
    the recording.
    """
    idx = err.get("word_index")
    if idx is not None and 0 <= idx < len(words):
        return {"start": words[idx]["start"], "end": words[idx]["end"]}
    return fallback.get(err["word"], {"start": None, "end": None})
