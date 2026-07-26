"""
Syllable stress placement analysis: compares each word's expected stress pattern
(from CMUdict, e.g. PREsent vs preSENT) against the stress detected in the audio.

PLACEHOLDER NOTICE: detecting actual stress placement from audio requires acoustic
analysis (pitch/energy per syllable) that is not yet integrated into this pipeline.
Until that's added, `_detect_stress()` below just returns the expected pattern
unchanged (i.e. assumes correct stress) — see the comment on that function.
"""

from typing import Dict, List, Optional

import pronouncing

from .phoneme_accuracy import clean_word


def get_expected_stress(word: str) -> Optional[str]:
    """Look up the expected stress pattern (e.g. '10' = primary-then-unstressed) via CMUdict."""
    cleaned = clean_word(word)
    phones = pronouncing.phones_for_word(cleaned)
    if not phones:
        return None
    return pronouncing.stresses(phones[0])


def _approximate_syllable_split(word: str, syllable_count: int) -> List[str]:
    """
    Rough heuristic: splits the word into `syllable_count` roughly-equal chunks.

    This is NOT real syllabification (that needs a dedicated hyphenation
    dictionary) — it's just good enough to render a readable stress-pattern
    string (e.g. "PREsent") for this prototype's UI.
    """
    if syllable_count <= 1 or syllable_count >= len(word):
        return [word]
    chunk_size = max(1, len(word) // syllable_count)
    chunks = [word[i : i + chunk_size] for i in range(0, len(word), chunk_size)]
    while len(chunks) > syllable_count:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return chunks


def render_stress_pattern(word: str, stress_digits: str) -> str:
    """Render a word with its stressed syllable capitalized, e.g. ('present', '01') -> 'preSENT'."""
    syllables = _approximate_syllable_split(word, len(stress_digits))
    primary_index = stress_digits.find("1")
    return "".join(
        syl.upper() if i == primary_index else syl.lower() for i, syl in enumerate(syllables)
    )


def _detect_stress(word: str, expected_stress: str) -> str:
    """
    PLACEHOLDER — no acoustic stress detection is integrated yet.

    Replace this with real pitch/energy analysis per syllable on the word's audio
    span — e.g. pitch tracking via `librosa.pyin` or CREPE, plus RMS energy per
    syllable, to find which syllable actually carries the stress peak. Needs a
    real syllable-boundary split too (see _approximate_syllable_split's own
    placeholder note above) to know which audio sub-span is which syllable.
    Until this is built, detected stress always equals expected, so
    stress_accuracy reads 1.0 for every word.
    """
    return expected_stress


def analyze_stress(words: List[Dict]) -> Dict:
    """
    Compare expected vs. detected syllable stress for each multi-syllable word.

    Input: [{"word": str, "start": float, "end": float}, ...]
    Output:
        {
            "stress_accuracy": float,
            "stress_errors": [{"word": str, "expected": str, "detected": str}, ...],
            "experimental": True,
                # Fix 8 (audio-pipeline correctness-fix plan): _detect_stress()
                # is a hard placeholder (always returns the expected pattern
                # unchanged — see its own PLACEHOLDER notice), so
                # stress_accuracy always reads 100% and stress_errors is
                # always empty. Flagged here so downstream consumers (e.g. a
                # coach-model training-dataset builder) can identify and
                # exclude this field programmatically, not just by convention.
        }
    """
    total = 0
    correct = 0
    stress_errors = []

    for w in words:
        cleaned = clean_word(w["word"])
        expected = get_expected_stress(cleaned)
        if expected is None or len(expected) <= 1:
            continue  # out-of-vocabulary or single-syllable word — no stress contrast to score

        detected = _detect_stress(cleaned, expected)
        total += 1
        if detected == expected:
            correct += 1
        else:
            stress_errors.append(
                {
                    "word": cleaned,
                    "expected": render_stress_pattern(cleaned, expected),
                    "detected": render_stress_pattern(cleaned, detected),
                }
            )

    accuracy = 1.0 if total == 0 else round(correct / total, 2)
    return {"stress_accuracy": accuracy, "stress_errors": stress_errors, "experimental": True}
