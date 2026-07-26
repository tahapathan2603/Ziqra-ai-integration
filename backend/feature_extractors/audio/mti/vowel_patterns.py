"""
Vowel pattern detection for MTI analysis: elongation, shortening, and substitution.

Reuses the same phoneme_errors data as the pronunciation module (no new audio
analysis), just filtered to vowel-to-vowel mismatches and classified by the kind
of vowel-quality shift involved. Vowel-quality tables live in
feature_extractors/audio/phoneme_patterns.py.
"""

from typing import Dict, List, Tuple

from ..phoneme_patterns import (
    GENERIC_VOWEL_SEVERITY,
    MTI_VOWEL_EXEMPT_WORDS,
    VOWEL_ELONGATION_PAIRS,
    VOWEL_PHONEMES,
    VOWEL_SHORTENING_PAIRS,
    severity_rank,
)
from .phoneme_substitution import build_word_timestamps, resolve_error_timestamp


def _classify_vowel_issue(expected: str, detected: str) -> Tuple[str, str]:
    """Returns (issue_type, severity) for a vowel-to-vowel mismatch."""
    if (expected, detected) in VOWEL_ELONGATION_PAIRS:
        return "vowel_elongation", VOWEL_ELONGATION_PAIRS[(expected, detected)]
    if (expected, detected) in VOWEL_SHORTENING_PAIRS:
        return "vowel_shortening", VOWEL_SHORTENING_PAIRS[(expected, detected)]
    return "vowel_substitution", GENERIC_VOWEL_SEVERITY


def analyze_vowel_patterns(phoneme_errors: List[Dict], words: List[Dict]) -> Dict:
    """
    Detect vowel elongation/shortening/substitution from phoneme-level errors.

    Input:
        phoneme_errors: [{"word": str, "expected": str, "detected": str}, ...]
        words: [{"word": str, "start": float, "end": float}, ...]

    Output:
        {
            "vowel_patterns": [
                {
                    "word": str,
                    "timestamp": {"start": float, "end": float},
                    "issue": str,       # vowel_elongation | vowel_shortening | vowel_substitution
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
            continue  # insertion/deletion, not a vowel quality shift
        if expected not in VOWEL_PHONEMES or detected not in VOWEL_PHONEMES:
            continue  # not a vowel-to-vowel mismatch
        if err["word"] in MTI_VOWEL_EXEMPT_WORDS:
            # Fix 3.4/6: schwa reduction and similar vowel-quality shifts on
            # true grammatical function words are ordinary connected speech,
            # not a native-language pattern (real observed damage: "the"
            # /ɪ/->/ə/ flagged as MTI). Deliberately a narrower set than
            # emphasis.py's FUNCTION_WORDS — see MTI_VOWEL_EXEMPT_WORDS'
            # own docstring for why (must not suppress "very" /v/->/w/, the
            # original motivating finding). Still counts toward pronunciation's
            # raw phoneme accuracy — this exclusion is MTI-specific.
            continue

        issue, severity = _classify_vowel_issue(expected, detected)
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
        # not word text, so a word genuinely mispronounced twice still counts
        # twice toward the recurrence threshold — which is exactly the evidence
        # a "recurring pattern" is supposed to be built from.
        key = (err["word"], err.get("word_index"))
        existing = by_word.get(key)
        if existing is None or severity_rank(severity) > severity_rank(existing["severity"]):
            by_word[key] = candidate

    return {"vowel_patterns": list(by_word.values())}
