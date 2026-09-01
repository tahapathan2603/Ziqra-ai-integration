"""
The quality gate must separate "this recording is bad" from "this speaker is bad".

Thresholds in quality.py are set between the two distributions measured on
this repo's own recordings: real Speech Accent Archive audio through consumer
mics, and copies of the same files with noise added at 5dB SNR. The numbers
in this file are those measurements, so a threshold change that stops
separating them fails here.
"""

from backend.api import reliability


def _level2_with_quality(verdict, stoi, pesq):
    return {
        "analysis": {
            "pronunciation": {"pronunciation_score": 88, "phoneme_accuracy": 0.9, "rhythm_score": 80},
            "intonation": {"intonation_score": 70, "monotonicity_score": 65},
            "engagement": {"engagement_score": 72},
            "fluency": {"words_per_minute": 130, "sentences_per_minute": 9, "filler_count": 2, "fillers_per_minute": 1},
        },
        "audio_quality": {"verdict": verdict, "stoi": stoi, "pesq": pesq},
    }


def test_poor_recording_suppresses_pronunciation_not_everything():
    level2 = _level2_with_quality("poor", 0.783, 1.118)  # the measured 5dB-SNR case
    report = reliability.apply(level2, speech_seconds=24.0, word_count=60)
    a = level2["analysis"]

    assert a["pronunciation"]["pronunciation_score"] is None, "noise looks exactly like poor pronunciation"
    assert a["pronunciation"]["phoneme_accuracy"] is None
    # Delivery survives: pace and pitch movement are still measurable through noise.
    assert a["fluency"]["words_per_minute"] == 130
    assert a["intonation"]["intonation_score"] == 70
    assert any("quality is poor" in r["reason"] for r in report["suppressed"])


def test_ok_recording_keeps_pronunciation():
    # A real phone recording: intelligible (high STOI) but lossy-coded, so a
    # low PESQ. This must NOT be called poor — every user records like this.
    level2 = _level2_with_quality("ok", 0.947, 1.502)
    reliability.apply(level2, speech_seconds=24.0, word_count=60)
    assert level2["analysis"]["pronunciation"]["pronunciation_score"] == 88


def test_unknown_quality_does_not_suppress():
    """Assessment failing must never cost the candidate their scores."""
    level2 = _level2_with_quality("unknown", None, None)
    reliability.apply(level2, speech_seconds=24.0, word_count=60)
    assert level2["analysis"]["pronunciation"]["pronunciation_score"] == 88


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
