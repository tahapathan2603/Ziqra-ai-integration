"""
Consonant pattern detection for MTI analysis: missing aspiration, retroflex
consonants, and general consonant substitutions.

Reuses the same phoneme_errors data as the pronunciation module (no new audio
analysis), filtered to consonant-to-consonant mismatches and classified by
pattern type. This is now the *single* place a consonant substitution is
classified and counted — the previously separate phoneme_substitution.py panel
double-counted known substitutions (e.g. /θ/->/t/) that are all consonant
shifts. Classification tables live in feature_extractors/audio/phoneme_patterns.py
so pronunciation's word-level severity agrees with the label shown here.
"""

from typing import Dict, List, Tuple

from ..phoneme_patterns import (
    CONSONANT_PHONEMES,
    CONSONANT_SUBSTITUTION_PATTERNS,
    GENERIC_CONSONANT_SEVERITY,
    severity_rank,
)
from .phoneme_substitution import build_word_timestamps, resolve_error_timestamp


def _classify_consonant_issue(expected: str, detected: str) -> Tuple[str, str]:
    """Returns (issue_type, severity) for a consonant-to-consonant mismatch."""
    pattern = CONSONANT_SUBSTITUTION_PATTERNS.get((expected, detected))
    if pattern is not None:
        return pattern
    return "consonant_substitution", GENERIC_CONSONANT_SEVERITY


def analyze_consonant_patterns(phoneme_errors: List[Dict], words: List[Dict]) -> Dict:
    """
    Detect consonant-level clarity issues from phoneme-level errors.

    Input:
        phoneme_errors: [{"word": str, "expected": str, "detected": str}, ...]
        words: [{"word": str, "start": float, "end": float}, ...]

    Output:
        {
            "consonant_patterns": [
                {
                    "word": str,
                    "timestamp": {"start": float, "end": float},
                    "issue": str,      # missing_aspiration | retroflex_consonant | consonant_substitution
                    "expected": str,
                    "detected": str,
                    "severity": str,
                },
                ...
            ]
        }
    """
    timestamps = build_word_timestamps(words)
    by_word: Dict[tuple, Dict] = {}

    for err in phoneme_errors:
        expected, detected = err["expected"], err["detected"]
        if expected is None or detected is None:
            continue  # insertion/deletion, not a consonant substitution
        if expected not in CONSONANT_PHONEMES or detected not in CONSONANT_PHONEMES:
            continue  # not a consonant-to-consonant mismatch
        # NOTE: no function-word exclusion here, deliberately — Fix 3.4/6's
        # rationale (schwa reduction is normal native-speaker behavior) is a
        # VOWEL-specific phenomenon (see vowel_patterns.py's
        # MTI_VOWEL_EXEMPT_WORDS use). A consonant substitution on any word,
        # function or not, is still a real, scorable finding — e.g. "very"
        # /v/->/w/ (the original motivating case) must not be suppressed just
        # because "very" happens to also appear on a stopword list built for
        # a different purpose (emphasis detection).

        issue, severity = _classify_consonant_issue(expected, detected)
        ts = resolve_error_timestamp(err, words, timestamps)
        candidate = {
            "word": err["word"],
            "timestamp": {"start": ts["start"], "end": ts["end"]},
            "issue": issue,
            "expected": expected,
            "detected": detected,
            "severity": severity,
        }

        # Fix 6: one word OCCURRENCE contributes at most one pattern entry
        # (real observed bug: "good" produced two entries from its two separate
        # phoneme errors). Keep the higher-severity one. Keyed by occurrence,
        # not word text — see vowel_patterns.py for the full rationale.
        key = (err["word"], err.get("word_index"))
        existing = by_word.get(key)
        if existing is None or severity_rank(severity) > severity_rank(existing["severity"]):
            by_word[key] = candidate

    return {"consonant_patterns": list(by_word.values())}
