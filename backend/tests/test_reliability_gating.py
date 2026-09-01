"""
Checks for the short-recording gate (backend/api/reliability.py).

The bug being pinned: a 4.5-second answer used to report intonation 96,
engagement 96 "highly engaging" and 83 words per minute — numbers produced by
analyzers that had found nothing to measure, then read aloud to the candidate
as coaching and handed to an LLM as evidence.
"""

from backend.api import reliability


def _level2(**overrides):
    analysis = {
        "fluency": {"words_per_minute": 83, "sentences_per_minute": 28, "filler_count": 0, "fillers_per_minute": 0},
        "pronunciation": {"pronunciation_score": 38, "phoneme_accuracy": 0.0, "rhythm_score": 0.77},
        "intonation": {"intonation_score": 96, "monotonicity_score": 100, "delivery_label": "Expressive"},
        "engagement": {"engagement_score": 96, "engagement_level": "highly_engaging"},
    }
    analysis.update(overrides)
    return {"analysis": analysis}


def test_short_answer_suppresses_unsupported_metrics():
    level2 = _level2()
    report = reliability.apply(level2, speech_seconds=3.8, word_count=11)
    a = level2["analysis"]

    assert a["intonation"]["intonation_score"] is None
    assert a["intonation"]["monotonicity_score"] is None
    assert a["engagement"]["engagement_score"] is None
    assert a["fluency"]["words_per_minute"] is None, "a rate from 3.8s of speech is extrapolation"
    assert a["fluency"]["filler_count"] == 0, "a count of what was heard is honest at any length"
    assert report["fully_measured"] is False
    assert report["suppressed"], "must say what it dropped and why"


def test_long_answer_keeps_everything():
    level2 = _level2()
    report = reliability.apply(level2, speech_seconds=24.4, word_count=61)
    a = level2["analysis"]

    assert a["intonation"]["intonation_score"] == 96
    assert a["engagement"]["engagement_score"] == 96
    assert a["fluency"]["words_per_minute"] == 83
    assert a["pronunciation"]["pronunciation_score"] == 38
    assert report["fully_measured"] is True
    assert report["suppressed"] == []


def test_pronunciation_gate_is_word_based():
    """Enough speech time but almost no words: phoneme accuracy is a ratio over
    aligned phonemes, so it needs words, not seconds."""
    level2 = _level2()
    reliability.apply(level2, speech_seconds=20.0, word_count=4)
    assert level2["analysis"]["pronunciation"]["phoneme_accuracy"] is None
    assert level2["analysis"]["pronunciation"]["pronunciation_score"] is None


def test_rhythm_is_rescaled_to_match_its_neighbours():
    level2 = _level2()
    reliability.rescale_rhythm(level2)
    assert level2["analysis"]["pronunciation"]["rhythm_score"] == 77, "0-1 published beside 0-100 scores"

    # Idempotent: a value already on the 0-100 scale must not be multiplied again.
    reliability.rescale_rhythm(level2)
    assert level2["analysis"]["pronunciation"]["rhythm_score"] == 77


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
