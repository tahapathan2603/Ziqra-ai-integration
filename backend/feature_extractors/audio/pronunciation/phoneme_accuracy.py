"""
Phoneme-level pronunciation accuracy: compares each word's expected phoneme
sequence (via espeak-ng G2P) against the phonemes actually detected in the
spoken audio (via a wav2vec2 CTC phoneme recognizer).

Both sides deliberately use the same phoneme inventory/notation (espeak's IPA
variant) so the comparison is meaningful — mixing a CMU-dict/ARPAbet-derived
"expected" side with an espeak-model "detected" side would compare two
different symbol systems, not real pronunciation differences.

Detection runs per SENTENCE, not per word. An earlier per-word version sliced
audio using each word's own Whisper timestamp — but on real (non-scripted)
speech those timestamps are frequently imprecise (observed durations from
0.11s to 0.72s for single words, some clearly bleeding into a neighboring word
or a pause) and short slices often gave wav2vec2 too little audio to produce
anything reliable at all. That noise swamped the real signal: phoneme_accuracy
read 0% on essentially every recording, correct or not. Running wav2vec2 once
per sentence gives it much more usable context and is far less sensitive to
any single word boundary being a little off; phonemes are then aligned back to
individual words via the sentence's own word-by-word phoneme breakdown (see
get_expected_phonemes_for_sentence).
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import torch
from phonemizer import phonemize
from phonemizer.separator import Separator
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from ..exclusions import compute_excluded_words
from ..phoneme_patterns import clean_word, is_coarticulation_artifact, salvage_phoneme, substitution_cost

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# Minimum audio needed to attempt phoneme decoding; anything shorter is
# unreliable to run through the model at all.
MIN_SLICE_SECONDS = 0.02

# Small fixed padding at each sentence's start/end so the first/last phoneme
# isn't clipped right at the boundary. Sentences (unlike individual words)
# usually have a real gap to their neighbor, so unlike the old per-word
# padding this doesn't need neighbor-aware capping.
#
# NOTE (found during the correctness-fix pass): this padding is NOT always
# enough. Observed real case: a sentence's own ASR-derived end timestamp
# (Whisper's sentence boundary, itself derived from its last word's timestamp)
# can slightly underestimate where that word actually finishes acoustically —
# the last real detected phoneme reached right up to the padded span's edge
# with no room left to decode the word's true remaining sounds. See
# _mark_trailing_clip_edge, which detects this from the alignment output
# directly (a run of deletions at the very tail of the expected sequence)
# rather than trying to guess a bigger padding value that still might not be
# enough.
SENTENCE_PADDING_SECONDS = 0.1

_processor = None
_model = None
_device = None


def _resolve_device() -> torch.device:
    """
    Prefer Apple Silicon GPU (MPS) over CPU — measured ~10x faster for this model
    (batched CPU inference: ~36s for 30 words; MPS: ~4s for the same batch, plus
    only ~6s to load and move the model). faster-whisper/ctranslate2 can't use
    MPS (see preprocessing/speech_to_text.py), but plain PyTorch/transformers
    models like this one can.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_model():
    """Load and cache the wav2vec2 phoneme recognizer (loaded once per process)."""
    global _processor, _model, _device
    if _model is None:
        logger.info(f"Loading phoneme recognition model ({MODEL_NAME})...")
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
        _model.eval()
        _device = _resolve_device()
        _model.to(_device)
        logger.info(f"Phoneme recognition model running on {_device}.")
    return _processor, _model, _device


def get_expected_phonemes(word: str) -> Optional[List[str]]:
    """Look up the expected IPA phoneme sequence for a single word via espeak-ng G2P (isolated/citation form)."""
    cleaned = clean_word(word)
    if not cleaned:
        return None
    result = phonemize(
        [cleaned], language="en-us", backend="espeak",
        separator=Separator(phone=" ", word=""), strip=True, njobs=1,
    )
    phonemes = result[0].split()
    return phonemes or None


def get_expected_phonemes_for_sentence(sentence_text: str) -> List[Optional[List[str]]]:
    """
    Phonemize a whole sentence in one espeak-ng call, returning one phoneme list
    per word (in the same order as sentence_text.split()).

    Phonemizing the sentence as a unit — rather than each word in isolation —
    gives natural connected-speech forms (e.g. "to" reduces to /tə/ in context
    instead of the citation form /tuː/), which is closer to how the word is
    actually said, while espeak's word separator still lets us recover which
    phonemes belong to which word.
    """
    words_in_text = sentence_text.split()
    if not words_in_text:
        return []

    result = phonemize(
        [sentence_text], language="en-us", backend="espeak",
        separator=Separator(phone=" ", word="|"), strip=True, njobs=1,
    )
    raw_groups = result[0].split("|")

    # Always return exactly len(words_in_text) entries, positionally aligned —
    # safer than filtering out empty groups, which could silently shift the
    # word<->phoneme mapping if some token phonemizes to nothing.
    groups: List[Optional[List[str]]] = []
    for i in range(len(words_in_text)):
        phonemes = raw_groups[i].strip().split() if i < len(raw_groups) else []
        groups.append(phonemes or None)
    return groups


def _slice_audio(waveform: torch.Tensor, sample_rate: int, start: float, end: float) -> torch.Tensor:
    start_sample = max(0, int(start * sample_rate))
    end_sample = min(waveform.shape[-1], int(end * sample_rate))
    return waveform[start_sample:end_sample]


def _collapse_ctc_with_timestamps(
    frame_ids: List[int], pad_id: int, time_per_frame: float, time_offset: float, vocab: Dict[int, str]
) -> List[Dict]:
    """
    Standard CTC collapse (drop blank tokens, merge consecutive repeats of the
    same token into one occurrence) — but unlike a plain decode, keep each
    collapsed occurrence's first/last frame index to derive a real timestamp.

    This reproduces exactly what Wav2Vec2Processor.batch_decode() does
    internally to the phoneme *sequence* (verified: produces an identical
    symbol sequence on real audio), it just additionally keeps the timing
    that decode() throws away.

    frame_ids must already be truncated to the sample's real (unpadded) frame
    count — see analyze_phoneme_accuracy's caller for why that matters for
    batched, variable-length inputs.

    Returns: [{"phoneme": str, "start": float, "end": float}, ...] in
    chronological order, with `time_offset` (the span's own start time in the
    full recording) already added in.
    """
    occurrences = []
    run_token = None
    run_start = None
    for i, tok in enumerate(frame_ids + [pad_id]):  # sentinel flush at the end
        if tok == run_token:
            continue
        if run_token is not None and run_token != pad_id:
            occurrences.append(
                {
                    "phoneme": f"/{vocab[run_token]}/",
                    "start": round(time_offset + run_start * time_per_frame, 3),
                    "end": round(time_offset + i * time_per_frame, 3),
                }
            )
        run_token, run_start = tok, i
    return occurrences


def _detect_phonemes_batch(
    waveform: torch.Tensor, sample_rate: int, spans: List[Tuple[float, float]]
) -> List[List[Dict]]:
    """
    Run the wav2vec2-espeak CTC phoneme recognizer on multiple audio spans in one
    batched forward pass, on MPS (Apple Silicon GPU) when available. Spans
    shorter than MIN_SLICE_SECONDS are skipped (empty result) without being sent
    through the model at all.

    Returns one list per span (in `spans` order) of
    {"phoneme": str, "start": float, "end": float} dicts, timestamped in
    absolute recording time (the span's own start offset is already applied).
    Real acoustic timestamps, not estimated — derived from which CTC output
    frame(s) each phoneme's token occupied, converted to time via the model's
    own (measured, not assumed) frame stride.
    """
    results: List[List[Dict]] = [[] for _ in spans]

    slices = []
    slice_indices = []
    slice_durations = []
    for i, (start, end) in enumerate(spans):
        audio_slice = _slice_audio(waveform, sample_rate, start, end)
        if audio_slice.numel() >= int(MIN_SLICE_SECONDS * sample_rate):
            slices.append(audio_slice.numpy())
            slice_indices.append(i)
            slice_durations.append(audio_slice.numel() / sample_rate)

    if not slices:
        return results

    processor, model, device = _get_model()
    inputs = processor(slices, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1).cpu()

    pad_id = processor.tokenizer.pad_token_id
    vocab = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    # Batched inputs are padded to the longest slice's length, so the logits
    # tensor's frame dimension is the same for every sample in the batch —
    # but a shorter slice's *real* (unpadded) frame count is smaller than
    # that. Using the padded frame count for every sample would read
    # meaningless trailing frames as extra (usually blank, but not always)
    # tokens for anything shorter than the longest slice in the batch.
    for local_i, (frame_ids_row, num_samples, span_idx) in enumerate(
        zip(predicted_ids.tolist(), (len(s) for s in slices), slice_indices)
    ):
        real_frame_count = model._get_feat_extract_output_lengths(num_samples).item()
        frame_ids = frame_ids_row[:real_frame_count]
        time_per_frame = slice_durations[local_i] / real_frame_count
        span_start = spans[span_idx][0]
        results[span_idx] = _collapse_ctc_with_timestamps(frame_ids, pad_id, time_per_frame, span_start, vocab)

    return results


def _align_phonemes(expected: List[str], detected: List[str]) -> List[Dict]:
    """
    Align expected vs. detected phoneme sequences by weighted edit distance and
    return the substitution/insertion/deletion differences, each tagged with
    the position in `expected` it corresponds to (for attributing the error
    back to whichever word owns that phoneme — see analyze_phoneme_accuracy).
    For insertions (extra detected phoneme with no expected counterpart), the
    position is the nearest surrounding expected index, since there's no exact
    word to blame for a phoneme that isn't supposed to exist at all.

    A naive position-by-position comparison breaks as soon as one extra or
    missing phoneme shifts everything after it — edit-distance alignment finds
    the actual correspondence, so a single real error doesn't cascade into
    every later phoneme reading as "wrong" too.

    Substitution cost is phonetically weighted (phoneme_patterns.substitution_cost)
    rather than a flat 1 — perceptually-equivalent pairs cost 0 (never a real
    difference) and same-class swaps cost less than cross-class ones. This
    matters for correctness, not just severity: with flat cost-1 substitutions,
    the DP is indifferent between "treat as one substitution" and "treat as one
    deletion + one insertion" (both cost 1 either way when a substitution
    doesn't reduce cost), so it can pick a path that fragments a real word's
    phonemes into insertions/deletions attributed elsewhere. Making a plausible
    substitution strictly cheaper than delete+insert (which now costs 2, not 1)
    biases the alignment toward keeping phonetically-related sounds matched to
    each other, reducing exactly this kind of cross-word misattribution.
    """
    n, m = len(expected), len(detected)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = substitution_cost(f"/{expected[i - 1]}/", f"/{detected[j - 1]}/")
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,  # substitution (0 if a true match)
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
            )

    errors = []
    i, j = n, m
    while i > 0 or j > 0:
        sub_cost = substitution_cost(f"/{expected[i - 1]}/", f"/{detected[j - 1]}/") if i > 0 and j > 0 else None
        if i > 0 and j > 0 and sub_cost == 0.0:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and abs(dp[i][j] - (dp[i - 1][j - 1] + sub_cost)) < 1e-9:
            errors.append(
                {"expected_index": i - 1, "expected": f"/{expected[i - 1]}/", "detected": f"/{detected[j - 1]}/"}
            )
            i, j = i - 1, j - 1
        elif i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1)) < 1e-9:
            errors.append({"expected_index": i - 1, "expected": f"/{expected[i - 1]}/", "detected": None})
            i -= 1
        else:
            errors.append({"expected_index": i, "expected": None, "detected": f"/{detected[j - 1]}/"})
            j -= 1
    errors.reverse()
    return errors


def _mark_clip_edge_runs(errors: List[Dict], expected_len: int) -> None:
    """
    Flag deletion errors that form a contiguous run at the very START or END
    of the sentence's expected phoneme sequence as clip_edge=True (mutates
    `errors` in place) — these reflect the decoded audio span ending before
    the word actually finished (or starting after it began) rather than a
    genuine mispronunciation.

    Real observed case: a sentence's own ASR-derived end timestamp
    underestimated where its last word ("industry") actually finished
    acoustically. wav2vec2 only decoded that word's first few phonemes before
    running out of audio in the padded span, so its true remaining phonemes
    came back as trailing deletions in the alignment. This is checked at the
    SEQUENCE level (not per-word) deliberately: whichever word the positional
    expected_index -> word mapping attributes these trailing indices to, a
    contiguous deletion run reaching the very last (or first) expected index
    is the acoustic-truncation signature regardless of exactly which word ends
    up named — see the audio-pipeline correctness-fix plan for the full
    diagnosis of why this is more robust than trusting the attribution alone.
    """
    if not errors or expected_len == 0:
        return

    by_index = {e["expected_index"]: e for e in errors if e["detected"] is None}

    idx = expected_len - 1
    while idx in by_index:
        by_index[idx]["clip_edge"] = True
        idx -= 1

    idx = 0
    while idx in by_index:
        by_index[idx]["clip_edge"] = True
        idx += 1


def _group_word_indices_by_sentence(
    words: List[Dict], sentences: List[Dict], tolerance: float = 1e-3
) -> List[List[int]]:
    """For each sentence, the indices into `words` whose start time falls within it."""
    groups: List[List[int]] = [[] for _ in sentences]
    for word_idx, w in enumerate(words):
        for sent_idx, sent in enumerate(sentences):
            if sent["start"] - tolerance <= w["start"] < sent["end"] + tolerance:
                groups[sent_idx].append(word_idx)
                break
    return groups


def analyze_phoneme_accuracy(
    words: List[Dict], sentences: List[Dict], waveform: torch.Tensor, sample_rate: int
) -> Dict:
    """
    Compare expected vs. detected phonemes, sentence by sentence, and score
    overall accuracy.

    Input:
        words: [{"word": str, "start": float, "end": float}, ...]
        sentences: [{"text": str, "start": float, "end": float}, ...] (from
            timestamps.process_timestamps()). If empty/None, the whole `words`
            span is treated as a single sentence.
        waveform: full recording as a mono torch.Tensor
        sample_rate: waveform's sample rate in Hz
    Output:
        {
            "phoneme_accuracy": float,
            "errors": [{"word": str, "word_index": int, "expected": str, "detected": str}, ...],
                # word_index = index into `words` identifying WHICH occurrence
                # of a repeated word this error belongs to, so consumers can
                # report per-occurrence severity/timestamps instead of merging
                # every occurrence of a word by its text.
                # Real, scorable errors only — clip-edge (audio-truncation)
                # artifacts and non-IPA garbage-token noise are excluded here
                # (and from phoneme_accuracy's denominator) rather than
                # reported as mispronunciations. See clip_edge_errors and
                # dropped_tokens below for what was excluded and why.
            "clip_edge_errors": [{"word": str, "expected": str}, ...],
                # Deletions at the very start/end of a sentence's expected
                # sequence — the decoded audio span ended before the word
                # actually finished (or started after it began), not a real
                # mispronunciation. See _mark_clip_edge_runs.
            "dropped_tokens": [{"phoneme": str, "start": float, "end": float}, ...],
                # Detected tokens outside the English IPA inventory that
                # couldn't be salvaged (see phoneme_patterns.salvage_phoneme) —
                # multilingual-model noise, never compared against expected
                # phonemes.
            "detected_phonemes": [{"phoneme": str, "start": float, "end": float}, ...]
                # Every phoneme wav2vec2 actually detected, in chronological
                # order, with real acoustic timestamps (see
                # _detect_phonemes_batch / _collapse_ctc_with_timestamps).
                # This is genuine ground truth (Level 1 material) — unlike
                # `errors`, which is an interpretation (a diff against
                # text-derived expected phonemes with no acoustic timing of
                # their own). Kept exactly as detected, never gated/salvaged —
                # gating only ever applies to the comparison copy.
            "skipped_words": [{"word": str, "reason": str}, ...],
                # Words excluded from scoring entirely (Fix 3) — proper names
                # and low-confidence/truncated fragments (see exclusions.py).
                # Their phonemes are removed from BOTH errors and the
                # phoneme_accuracy denominator, not just from the error count.
            "coarticulation_matches": [{"word": str, "expected": str, "detected": str}, ...],
                # Substitutions explained by ordinary connected-speech
                # coarticulation (Fix 6) — treated as correct (counted in the
                # accuracy denominator, not as an error), never a native-
                # language pattern. See phoneme_patterns.is_coarticulation_artifact.
        }
    """
    if not sentences:
        sentences = (
            [{"text": " ".join(w["word"] for w in words), "start": words[0]["start"], "end": words[-1]["end"]}]
            if words
            else []
        )

    word_groups = _group_word_indices_by_sentence(words, sentences)
    recording_duration = waveform.shape[-1] / sample_rate

    # Fix 3: words that must never be scored (proper names, low-confidence/
    # truncated fragments) — computed once up front from `words` alone, no
    # acoustic data needed. Applied below by removing their phonemes from both
    # the error list AND the accuracy denominator (not just the error count),
    # so excluding a word can't silently inflate accuracy either.
    excluded = compute_excluded_words(words, recording_duration)

    # Resolve each sentence's per-word expected phonemes and audio span first,
    # so acoustic detection can run as a single batched call across all
    # sentences (one wav2vec2 call for the whole recording, not one per word
    # and not even one per sentence individually).
    sentence_entries = []  # (word_indices, expected_seq, phoneme_to_local_word)
    spans = []
    for sentence, word_indices in zip(sentences, word_groups):
        if not word_indices:
            continue

        sentence_text = sentence.get("text") or " ".join(words[i]["word"] for i in word_indices)
        per_word_phonemes = get_expected_phonemes_for_sentence(sentence_text)

        expected_seq: List[str] = []
        phoneme_to_local_word: List[int] = []
        for local_idx, phonemes in enumerate(per_word_phonemes):
            if phonemes:
                expected_seq.extend(phonemes)
                phoneme_to_local_word.extend([local_idx] * len(phonemes))

        if not expected_seq:
            continue

        sentence_entries.append((word_indices, expected_seq, phoneme_to_local_word))
        padded_start = max(0.0, sentence["start"] - SENTENCE_PADDING_SECONDS)
        padded_end = min(recording_duration, sentence["end"] + SENTENCE_PADDING_SECONDS)
        spans.append((padded_start, padded_end))

    detected_batch = _detect_phonemes_batch(waveform, sample_rate, spans)

    total_phonemes = 0
    mismatched_phonemes = 0
    errors = []
    clip_edge_errors = []
    coarticulation_matches: List[Dict] = []
    dropped_tokens: List[Dict] = []
    detected_phonemes: List[Dict] = []
    for (word_indices, expected_seq, phoneme_to_local_word), detected_entries in zip(sentence_entries, detected_batch):
        # Vocab gate (Fix 2): salvage what's recoverable (tone-digit-tagged
        # multilingual noise like "/a5/" -> "/a/"), drop what isn't. Gating
        # applies ONLY to this comparison copy — detected_phonemes (Level 1
        # ground truth, below) always keeps the raw, unmodified stream.
        detected_seq: List[str] = []
        for d in detected_entries:
            salvaged = salvage_phoneme(d["phoneme"])
            if salvaged is not None:
                detected_seq.append(salvaged.strip("/"))
            else:
                dropped_tokens.append({"phoneme": d["phoneme"], "start": d["start"], "end": d["end"]})

        sentence_errors = _align_phonemes(expected_seq, detected_seq)
        _mark_clip_edge_runs(sentence_errors, len(expected_seq))

        # Fix 3: which local words in THIS sentence are excluded, and how many
        # expected phonemes each one contributes — needed to subtract their
        # full contribution from the denominator below, not just skip their
        # errors (skipping only the errors would inflate accuracy for free).
        excluded_local_words = {
            local_idx
            for local_idx in set(phoneme_to_local_word)
            if clean_word(words[word_indices[local_idx]]["word"]) in excluded
        }
        excluded_phoneme_count = sum(1 for local_idx in phoneme_to_local_word if local_idx in excluded_local_words)

        for err in sentence_errors:
            local_idx = phoneme_to_local_word[min(err["expected_index"], len(phoneme_to_local_word) - 1)]
            if local_idx in excluded_local_words:
                continue  # proper name / low-confidence — never a scored error
            word_index = word_indices[local_idx]
            word = words[word_index]
            # word_index (index into `words`) identifies WHICH OCCURRENCE of a
            # repeated word this error belongs to. Without it, consumers can
            # only key by word text, which silently merges every occurrence of
            # a common word: three separate "a"s with one slip each collapsed
            # into a single entry with three errors and read as "high severity
            # mispronunciation", stamped with the first occurrence's timestamp.
            entry = {
                "word": clean_word(word["word"]),
                "word_index": word_index,
                "expected": err["expected"],
                "detected": err["detected"],
            }

            if err.get("clip_edge"):
                clip_edge_errors.append(entry)
                continue

            # Fix 6: coarticulation (connected-speech effects at word
            # boundaries, e.g. "about yourself" -> yod-coalescence to /tʃ/) is
            # universal across English speakers, never a native-language
            # pattern — treat as a match (counts toward total_phonemes below,
            # same as any correctly-pronounced phoneme) rather than an error,
            # so it never reaches pronunciation's mispronounced_words OR MTI
            # (which both consume this same errors list).
            following_idx = err["expected_index"] + 1
            following_expected = f"/{expected_seq[following_idx]}/" if following_idx < len(expected_seq) else None
            if err["expected"] is not None and err["detected"] is not None and is_coarticulation_artifact(
                err["expected"], err["detected"], following_expected
            ):
                coarticulation_matches.append(entry)
                continue

            errors.append(entry)
            mismatched_phonemes += 1

        # Clip-edge and excluded-word positions are both removed from the
        # accuracy denominator, not just from the error count — otherwise
        # excluding a word's errors would inflate accuracy for free, and
        # counting undecodable clip-edge phonemes as "expected but unscored"
        # would deflate it, for the same reason.
        clip_edge_count = sum(
            1 for e in sentence_errors
            if e.get("clip_edge")
            and phoneme_to_local_word[min(e["expected_index"], len(phoneme_to_local_word) - 1)] not in excluded_local_words
        )
        total_phonemes += len(expected_seq) - clip_edge_count - excluded_phoneme_count

        detected_phonemes.extend(detected_entries)

    accuracy = 1.0 if total_phonemes == 0 else round(max(0.0, 1 - (mismatched_phonemes / total_phonemes)), 2)
    skipped_words = [{"word": word_text, "reason": reason} for word_text, reason in excluded.items()]

    return {
        "phoneme_accuracy": accuracy,
        "errors": errors,
        "clip_edge_errors": clip_edge_errors,
        "coarticulation_matches": coarticulation_matches,
        "dropped_tokens": dropped_tokens,
        "detected_phonemes": detected_phonemes,
        "skipped_words": skipped_words,
    }
