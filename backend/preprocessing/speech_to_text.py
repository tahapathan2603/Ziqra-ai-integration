"""Speech-to-text: transcribe VAD-generated speech chunks into a full transcript."""

import logging
from typing import Dict, List, Tuple

import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

MODEL_SIZE = "large-v3"
DEVICE = "auto"
# float16 on GPU, int8 on CPU. "default" resolves to the converted model's own
# dtype — float32 for large-v3 — which is roughly twice the compute and twice
# the VRAM for no accuracy gain that matters here.
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

_model = None


def get_model() -> WhisperModel:
    """
    Load and cache the Whisper model (loaded once per process).

    Public so other preprocessing modules (e.g. timestamps.py) can reuse the same
    cached model instance instead of loading a second multi-GB copy into memory.
    """
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model ({MODEL_SIZE})...")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def transcribe_chunk(chunk_path: str, word_timestamps: bool = False) -> Tuple[str, List[Dict], Dict]:
    """
    Transcribe a single audio chunk file.

    word_timestamps=True computes per-word timing in the same decoding pass
    (Whisper derives it from attention weights it already has to compute) —
    much cheaper than what timestamps.py used to do, which was transcribing
    the same audio a second time from scratch just to get word timing.

    Returns (text, words, language_info). `words` is [] unless
    word_timestamps=True; timestamps in `words` are chunk-relative (caller
    offsets to absolute recording time). Each word also carries `confidence`
    (faster-whisper's own per-word probability — real model output, not a
    placeholder). `language_info` is {"language": str, "language_probability":
    float}, Whisper's own coarse (ISO-639-1, e.g. "en") detection — it cannot
    and does not detect regional/accent variants (e.g. "en-IN"); do not
    upgrade this to a regional label without a dedicated model, since that
    would also cross into the accent-classification this project's MTI module
    explicitly avoids (see feature_extractors/audio/mti/mti_analyzer.py).
    """
    model = get_model()
    segments, info = model.transcribe(
        chunk_path,
        beam_size=5,
        word_timestamps=word_timestamps,
        # Whisper's well-known failure on short or near-silent audio is to
        # invent a stock phrase — "Thank you.", "Thanks for watching!" — and a
        # VAD chunk of a hesitant answer is exactly that kind of input. A real
        # production answer came back as "I am Dave. Well, thank you." for
        # this reason, and because the transcript is the reference the
        # pronunciation scorer compares against, an invented tail does not
        # just read badly: it drags the phoneme accuracy down with it.
        #
        # condition_on_previous_text=False stops one chunk's hallucination
        # seeding the next; the two thresholds make the decoder drop a segment
        # it is not reasonably confident about rather than emit filler.
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        temperature=[0.0, 0.2, 0.4],
    )

    text_parts = []
    words = []
    for segment in segments:
        text_parts.append(segment.text)
        if word_timestamps:
            for word in segment.words or []:
                words.append(
                    {
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                        "confidence": round(word.probability, 4),
                    }
                )

    language_info = {"language": info.language, "language_probability": round(info.language_probability, 4)}
    return "".join(text_parts).strip(), words, language_info


def transcribe_chunks(chunks: List[Dict], word_timestamps: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Transcribe each speech chunk.

    Input: [{"chunk_id": int, "path": str, "start": float, "end": float}, ...]
    Returns (transcripts, language_info):
        transcripts: [{"chunk_id": int, "start": float, "end": float, "text": str}, ...]
            (each entry also gets a "words" key, offset to absolute recording
            time, when word_timestamps=True)
        language_info: {"language": str, "language_probability": float} —
            Whisper's own detection from the first chunk (a session is a
            single speaker/language, so one detection is representative;
            re-detecting per chunk would be redundant work for no real gain).
    """
    transcripts = []
    language_info = {"language": None, "language_probability": None}
    for i, chunk in enumerate(chunks):
        logger.info(f"Transcribing chunk {chunk['chunk_id']}...")
        text, words, chunk_language_info = transcribe_chunk(chunk["path"], word_timestamps=word_timestamps)
        if i == 0:
            language_info = chunk_language_info
        entry = {
            "chunk_id": chunk["chunk_id"],
            "start": chunk["start"],
            "end": chunk["end"],
            "text": text,
        }
        if word_timestamps:
            offset = chunk["start"]
            entry["words"] = [
                {"word": w["word"], "start": offset + w["start"], "end": offset + w["end"], "confidence": w["confidence"]}
                for w in words
            ]
        transcripts.append(entry)
    return transcripts, language_info


def merge_transcripts(transcripts: List[Dict]) -> str:
    """Merge ordered per-chunk transcripts into a single full transcript string."""
    logger.info("Merging transcripts...")
    ordered = sorted(transcripts, key=lambda t: t["chunk_id"])
    return " ".join(t["text"] for t in ordered if t["text"]).strip()


def process_chunks(chunks: List[Dict], word_timestamps: bool = False) -> Dict:
    """
    Run the full STT pipeline: transcribe each chunk, then merge into a full transcript.

    word_timestamps=True additionally computes word-level timing (each word
    also carrying a real `confidence`) in the same pass and includes it as a
    top-level "words" key — pass it through to timestamps.process_timestamps()
    to skip its own (otherwise redundant) transcription pass. Default is
    False, so existing callers are unaffected.

    Returns:
        {
            "full_transcript": str,
            "segments": [{"chunk_id": int, "start": float, "end": float, "text": str}, ...],
            "words": [{"word": str, "start": float, "end": float, "confidence": float}, ...]  # only if word_timestamps=True
            "detected_language": str,        # e.g. "en" — Whisper's own coarse detection
            "language_probability": float,
        }
    """
    transcripts, language_info = transcribe_chunks(chunks, word_timestamps=word_timestamps)
    full_transcript = merge_transcripts(transcripts)

    result = {
        "full_transcript": full_transcript,
        "segments": [{k: v for k, v in t.items() if k != "words"} for t in transcripts],
        "detected_language": language_info["language"],
        "language_probability": language_info["language_probability"],
    }
    if word_timestamps:
        result["words"] = [w for t in transcripts for w in t.get("words", [])]
    return result
