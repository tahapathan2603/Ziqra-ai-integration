"""
Sentence segmentation must survive an ASR that forgets to punctuate.

Measured on real unscripted recordings, Whisper large-v3 returned two of five
with no terminal punctuation anywhere. The whole answer then became a single
"sentence", which slices the entire utterance into wav2vec2 in one go and
reports a confident sentences-per-minute of roughly nothing.
"""

# timestamps.py imports speech_to_text, which imports faster_whisper (a GPU
# dependency that isn't installed outside the Modal image). Segmentation is
# pure timing arithmetic and needs none of it, so the module is stubbed rather
# than making this test require a multi-gigabyte install to run.
import sys
import types

if "faster_whisper" not in sys.modules:
    stub = types.ModuleType("faster_whisper")
    stub.WhisperModel = object
    sys.modules["faster_whisper"] = stub
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub

from backend.preprocessing.timestamps import build_sentence_timestamps


def _words(spec):
    """spec: [(word, start, end), ...]"""
    return [{"word": w, "start": s, "end": e, "confidence": 0.9} for w, s, e in spec]


def test_punctuation_is_used_when_present():
    words = _words([("Hello.", 0.0, 0.4), ("I", 0.5, 0.6), ("am", 0.7, 0.9), ("Asha.", 1.0, 1.5)])
    sentences = build_sentence_timestamps(words)
    assert len(sentences) == 2


def test_unpunctuated_run_on_is_split_on_pauses():
    spec = []
    t = 0.0
    # 14 words in three bursts separated by ~1s of silence, no punctuation.
    for burst in range(3):
        for i in range(5 if burst < 2 else 4):
            spec.append((f"word{burst}{i}", t, t + 0.3))
            t += 0.35
        t += 1.0
    sentences = build_sentence_timestamps(_words(spec))
    assert len(sentences) == 3, f"expected one sentence per burst, got {len(sentences)}"


def test_short_unpunctuated_answer_is_left_alone():
    """Below the word floor a run-on is not worth splitting — and a genuinely
    one-sentence answer must not be chopped at every breath."""
    spec = [(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(6)]
    assert len(build_sentence_timestamps(_words(spec))) == 1


def test_pause_fallback_does_not_fire_when_any_punctuation_exists():
    spec = [(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(13)]
    words = _words(spec)
    words[-1]["word"] = "end."
    # One long gap, but the transcript is punctuated, so punctuation wins.
    words[7]["start"] += 2.0
    words[7]["end"] += 2.0
    assert len(build_sentence_timestamps(words)) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
