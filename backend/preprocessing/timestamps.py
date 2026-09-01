"""Timestamp extraction: word- and sentence-level timestamps for the transcript."""

import logging
from typing import Dict, List, Optional

try:
    # Works when imported as part of the backend.preprocessing package
    # (e.g. `from backend.preprocessing import timestamps`).
    from .speech_to_text import get_model
except ImportError:
    # Works when imported as a bare sibling module, which is how silero_vad.py's
    # __main__ block reaches it when this file is run directly as a script.
    from speech_to_text import get_model

logger = logging.getLogger(__name__)

SENTENCE_END_CHARS = (".", "!", "?")


def extract_word_timestamps(chunks: List[Dict]) -> List[Dict]:
    """
    Run word-level alignment (via Whisper's word_timestamps) on each chunk's audio,
    offsetting timestamps by the chunk's absolute start time in the full recording.

    Fallback path only — this re-transcribes every chunk from scratch just to
    get word timing, which is redundant if speech_to_text.process_chunks() was
    already called with word_timestamps=True (see process_timestamps() below,
    which prefers pre-computed words and only falls back to this when none were
    provided, e.g. when testing this module in isolation).

    Input: [{"chunk_id": int, "path": str, "start": float, "end": float}, ...]
    Output: [{"word": str, "start": float, "end": float, "confidence": float}, ...]
        in chronological order.
    """
    model = get_model()
    words = []

    for chunk in sorted(chunks, key=lambda c: c["chunk_id"]):
        logger.info(f"Extracting word timestamps for chunk {chunk['chunk_id']}...")
        segments, _info = model.transcribe(chunk["path"], word_timestamps=True, beam_size=5)
        offset = chunk["start"]

        for segment in segments:
            for word in segment.words or []:
                words.append(
                    {
                        "word": word.word.strip(),
                        "start": offset + word.start,
                        "end": offset + word.end,
                        "confidence": round(word.probability, 4),
                    }
                )

    return words


def _finalize_sentence(sentence_words: List[Dict]) -> Dict:
    return {
        "text": " ".join(w["word"] for w in sentence_words),
        "start": sentence_words[0]["start"],
        "end": sentence_words[-1]["end"],
    }


# A pause this long between words ends a sentence when the ASR gave us no
# punctuation to go on. 0.7s is comfortably longer than the gaps inside a
# phrase and shorter than the beat between two spoken sentences.
PAUSE_SENTENCE_BREAK_SECONDS = 0.7
# Below this many words, a run-on is not worth splitting; above it, treating
# the whole answer as one "sentence" starts to matter.
MIN_WORDS_BEFORE_PAUSE_SPLIT = 12


def build_sentence_timestamps(words: List[Dict]) -> List[Dict]:
    """
    Group words into sentences using terminal punctuation (./!/?), deriving each
    sentence's start/end from its first and last word.

    Falls back to pauses when the transcript has no terminal punctuation at
    all. Whisper usually punctuates, but not always — measured on real
    unscripted recordings, two of five came back as a single unpunctuated
    run-on ("I love the dynamic fast-paced nature of sales and the fact that
    my hard work ... " with no stops anywhere).

    That silently degrades more than it looks: the whole answer becomes one
    "sentence", so pronunciation slices the entire utterance into wav2vec2 in
    one go instead of sentence by sentence, and sentences-per-minute reports
    a confident 2.5 for a two-minute answer. Timing is independent of the
    ASR's punctuation habits, so it is the more reliable signal when
    punctuation is missing entirely.
    """
    logger.info("Building sentence timestamps...")
    sentences = []
    current: List[Dict] = []

    for word in words:
        current.append(word)
        if word["word"].endswith(SENTENCE_END_CHARS):
            sentences.append(_finalize_sentence(current))
            current = []

    if current:
        sentences.append(_finalize_sentence(current))

    if _needs_pause_fallback(words, sentences):
        logger.info("No terminal punctuation in the transcript — segmenting on pauses instead.")
        return _segment_on_pauses(words)

    return sentences


def _needs_pause_fallback(words: List[Dict], sentences: List[Dict]) -> bool:
    """True when punctuation produced a single run-on out of a long transcript."""
    return (
        len(sentences) <= 1
        and len(words) >= MIN_WORDS_BEFORE_PAUSE_SPLIT
        and not any(w["word"].endswith(SENTENCE_END_CHARS) for w in words)
    )


def _segment_on_pauses(words: List[Dict]) -> List[Dict]:
    sentences = []
    current: List[Dict] = []
    for previous, word in zip([None] + words[:-1], words):
        if (
            current
            and previous is not None
            and (word.get("start", 0.0) - previous.get("end", 0.0)) >= PAUSE_SENTENCE_BREAK_SECONDS
        ):
            sentences.append(_finalize_sentence(current))
            current = []
        current.append(word)
    if current:
        sentences.append(_finalize_sentence(current))
    return sentences


def build_transcript_structure(stt_result: Dict, sentences: List[Dict], words: List[Dict]) -> Dict:
    """Assemble the unified transcript structure: transcript + segments + sentences + words."""
    return {
        "full_transcript": stt_result["full_transcript"],
        "segments": stt_result["segments"],
        "sentences": sentences,
        "words": words,
        # Passed straight through from speech_to_text.process_chunks() — no
        # logic here, just plumbing so callers get language info from the
        # same unified structure as everything else.
        "detected_language": stt_result.get("detected_language"),
        "language_probability": stt_result.get("language_probability"),
    }


def process_timestamps(stt_result: Dict, chunks: List[Dict], words: Optional[List[Dict]] = None) -> Dict:
    """
    Run the full timestamp pipeline: derive sentence timestamps from word
    timestamps, and assemble the unified transcript structure.

    Args:
        stt_result: output of speech_to_text.process_chunks() -> {"full_transcript", "segments"}
        chunks: the same chunk records passed into speech_to_text.process_chunks()
                -> [{"chunk_id": int, "path": str, "start": float, "end": float}, ...]
        words: pre-computed word timestamps, e.g.
            speech_to_text.process_chunks(chunks, word_timestamps=True)["words"] —
            pass this to avoid a second Whisper pass over the same audio. Falls
            back to extract_word_timestamps(chunks) (a fresh transcription pass)
            if not provided.
    """
    if words is None:
        words = extract_word_timestamps(chunks)
    sentences = build_sentence_timestamps(words)
    return build_transcript_structure(stt_result, sentences, words)
