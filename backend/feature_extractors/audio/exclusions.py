"""
Unified word-exclusion mechanism (Fix 3 + Fix 9.3 of the audio-pipeline
correctness-fix plan).

Certain transcript tokens must never be scored as pronunciation/MTI findings,
because doing so produces confident-sounding false positives that would then
become training labels:

1. proper_name — a person's own name isn't real vocabulary; G2P and phoneme
   comparison against it is meaningless. Real observed damage: "samad" and
   "periwal" were G2P'd as English words and flagged high-severity, dragging
   phoneme accuracy to 69% and MTI to "high impact" single-handedly.
2. low_confidence — ASR itself is unsure this word is even right. Real observed
   damage: "Thank" at the very start of a recording (ASR confidence 0.137, only
   2 phonemes decoded) was actually the tail of the interview PROMPT bleeding in
   before the candidate's real answer starts, not a mispronunciation by the
   candidate.
3. clip_edge — the word sits within CLIP_EDGE_SECONDS of the recording's start
   or end, so it may be acoustically truncated no matter how confident ASR is.
   Kept as its own reason rather than folded into low_confidence: measured on
   real audio, a word transcribed at 99.5% confidence was excluded solely for
   starting 0.2s into the clip, and labelling that "low_confidence" is simply
   untrue — these reasons become training data.
4. (reserved) prompt / excluded_speaker — Fix 9's prompt-contamination and
   speaker-diarization exclusions. Not implemented (no prompt metadata reaches
   this pipeline today) — the reason value exists so this module's shape
   doesn't need to change again when that's built.

This word-level clip_edge gate is deliberately COARSER than, and complementary
to, phoneme_accuracy.py's per-SENTENCE clip-edge detection (Fix 1): that one
catches individual phonemes falling outside the decoded span, while this one
drops a whole word whose audio may be cut off at the recording boundary.

NOTE (tuning caveat): on already-chunked audio (a VAD chunk fed in directly as
its own recording), speech starts almost immediately, so CLIP_EDGE_SECONDS can
exclude genuine opening content — measured: a clean 99.5%-confidence word at
0.2s. That is correct behaviour for a truncated clip but over-eager for a
deliberately pre-trimmed one; revisit the threshold if pre-chunked input
becomes a normal path rather than a testing convenience.

NOTE (honest limitation on proper-name detection): word-frequency alone does
NOT cleanly separate real names from genuinely rare English words —
empirically, "samad" (zipf 2.21) sits in the same frequency band as real
words like "onomatopoeia" (2.15) and "quixotic" (2.33). The cue-phrase check
below is what actually catches the common, demonstrated case (interview
self-introductions: "my name is X", "I'm X"); the bare frequency threshold is
a conservative supplementary signal, not a reliable standalone detector. See
the audio-pipeline correctness-fix plan for the calibration data.

NOTE (honest limitation on scope): exclusions are matched by cleaned word TEXT,
not by transcript position — consistent with word_pronunciation.py's own
existing "first-occurrence timestamp per word" simplification (see that
module's docstring), not a new limitation introduced here. This means if the
same word text appears more than once with different confidence/context, ALL
occurrences share the same excluded/included verdict. Acceptable for the
demonstrated real cases (a stray truncated prompt-fragment word, or a proper
name) — both are typically unique, non-repeating tokens in a short response.
"""

from typing import Dict, List

from wordfreq import zipf_frequency

from .phoneme_patterns import clean_word

# Below this zipf frequency, a word is rare enough to be a plausible name/OOV
# candidate even without a cue phrase — deliberately conservative (very low)
# since frequency alone can't reliably separate names from rare real words;
# see module docstring.
NAME_ZIPF_THRESHOLD = 0.5

# Phrases that, in an interview self-introduction, are almost always
# immediately followed by the candidate's own name.
NAME_CUE_PHRASES = ("name is", "i'm", "i am", "this is", "call me")

# Up to this many tokens after a cue phrase are treated as name candidates
# (covers first + last name), stopping early at the first function word.
NAME_SPAN_MAX_TOKENS = 2

MIN_ASR_CONFIDENCE = 0.5
CLIP_EDGE_SECONDS = 0.3


def _function_words():
    # Local import: phoneme_patterns already imports from this package's
    # pronunciation subpackage in a couple of places; importing FUNCTION_WORDS
    # lazily here avoids adding a module-load-order dependency for something
    # only needed inside one function.
    from .phoneme_patterns import FUNCTION_WORDS
    return FUNCTION_WORDS


def _name_span_after_cue(cleaned_words: List[str], cue_end_idx: int) -> List[int]:
    """Indices of up to NAME_SPAN_MAX_TOKENS words starting at cue_end_idx,
    stopping early at a function word (which would signal the name has ended,
    e.g. "...samad periwal i study..." stops before "i")."""
    function_words = _function_words()
    indices = []
    for offset in range(NAME_SPAN_MAX_TOKENS):
        idx = cue_end_idx + offset
        if idx >= len(cleaned_words):
            break
        if cleaned_words[idx] in function_words:
            break
        indices.append(idx)
    return indices


def _find_name_cue_indices(cleaned_words: List[str]) -> List[int]:
    """Indices of words immediately following a NAME_CUE_PHRASES match."""
    name_indices: List[int] = []
    for phrase in NAME_CUE_PHRASES:
        phrase_tokens = phrase.split()
        plen = len(phrase_tokens)
        for start in range(len(cleaned_words) - plen + 1):
            if cleaned_words[start:start + plen] == phrase_tokens:
                name_indices.extend(_name_span_after_cue(cleaned_words, start + plen))
    return name_indices


def compute_excluded_words(words: List[Dict], total_duration: float) -> Dict[str, str]:
    """
    Returns {cleaned_word_text: reason} for every word that must be excluded
    from pronunciation/MTI scoring — reason is one of "proper_name",
    "low_confidence" (ASR itself is unsure of the word), or "clip_edge" (the
    word sits in the first/last CLIP_EDGE_SECONDS, so it may be acoustically
    truncated regardless of how confident ASR is). See module docstring for
    detection method and honest limitations (matched by word text, not
    transcript position).

    Args:
        words: [{"word": str, "start": float, "end": float, "confidence": float}, ...]
               in chronological order (Level 1 timeline shape).
        total_duration: full recording duration in seconds.
    """
    if not words:
        return {}

    cleaned_words = [clean_word(w["word"]) for w in words]
    excluded: Dict[str, str] = {}

    # --- proper names: cue-phrase context (primary) + rarity (supplementary) ---
    for idx in _find_name_cue_indices(cleaned_words):
        word_text = cleaned_words[idx]
        if word_text:
            excluded[word_text] = "proper_name"

    for word_text in cleaned_words:
        if word_text and word_text not in excluded and zipf_frequency(word_text, "en") < NAME_ZIPF_THRESHOLD:
            excluded[word_text] = "proper_name"

    # --- low confidence / truncated fragments ---
    # The two rules report DISTINCT reasons. Collapsing both into
    # "low_confidence" produced flatly false labels: measured on real audio, a
    # word transcribed at 99.5% confidence was excluded purely for starting
    # 0.2s into the clip and still reported as "low_confidence". These reasons
    # end up in training data and in the coach packets, so a wrong one is worse
    # than no label at all. Genuine low confidence takes precedence when both
    # apply, since it's the stronger signal that the word itself is unreliable.
    for w, word_text in zip(words, cleaned_words):
        if not word_text or word_text in excluded:
            continue
        confidence = w.get("confidence")
        touches_edge = w["start"] < CLIP_EDGE_SECONDS or w["end"] > total_duration - CLIP_EDGE_SECONDS
        if confidence is not None and confidence < MIN_ASR_CONFIDENCE:
            excluded[word_text] = "low_confidence"
        elif touches_edge:
            excluded[word_text] = "clip_edge"

    return excluded
