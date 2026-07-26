"""
Shared phoneme classification tables for the audio analysis modules.

Both the pronunciation module (word-level severity in word_pronunciation.py) and
the MTI module (pattern classification in vowel_patterns.py /
consonant_patterns.py) classify the same phoneme errors. Keeping one table here
guarantees a given error is always labeled and graded identically everywhere it
appears — previously each module kept its own copy and the same error could
read "high" severity in one panel and "medium" in another, or be counted twice
(once as a generic "phoneme substitution", once as a consonant pattern).

Pure data module: no imports from the pronunciation or mti packages, so either
side can depend on it without cycles. Phoneme notation matches
pronunciation/phoneme_accuracy.py's error entries: slash-wrapped IPA, e.g. "/θ/".
"""

import re as _word_re


def clean_word(word: str) -> str:
    """Strip punctuation (Whisper attaches trailing commas/periods etc.) and
    lowercase. Lives here (not in phoneme_accuracy.py, which re-imports it for
    backward compatibility) specifically so exclusions.py can use it without
    creating a cycle: phoneme_accuracy.py depends on exclusions.py (Fix 3), so
    exclusions.py cannot depend back on phoneme_accuracy.py."""
    return _word_re.sub(r"[^a-zA-Z']", "", word).lower()

VOWEL_PHONEMES = {
    "/a/", "/aɪ/", "/aʊ/", "/aː/", "/ã/",
    "/e/", "/eɪ/", "/ɛ/", "/ɜː/",
    "/ɪ/", "/i/", "/iː/",
    "/ɔ/", "/ɔː/", "/ɔɪ/",
    "/oʊ/", "/ʊ/", "/uː/", "/u/",
    "/ʌ/", "/ə/", "/æ/", "/ɑː/", "/ɑ/",
    # R-colored (rhotic) vowels and syllabic consonants — espeak emits these as
    # single vowel-nucleus units in American English (e.g. "pharma" -> .../ɑːɹ/
    # /m/ /ɚ/, "little" -> .../t/ /əl/). Confirmed present in real detected
    # streams (see Fix 2 vocab-gate work) — omitting them previously meant any
    # word containing one could never validate, since the symbol itself wasn't
    # recognized as phonemic content.
    "/ɑːɹ/", "/ɔːɹ/", "/ɚ/", "/əl/",
    # espeak's reduced/near-close central vowel (unstressed "-es"/"-ed" etc.).
    "/ᵻ/",
}

CONSONANT_PHONEMES = {
    "/p/", "/b/", "/t/", "/d/", "/k/", "/ɡ/", "/g/",
    "/tʃ/", "/dʒ/", "/f/", "/v/", "/θ/", "/ð/",
    "/s/", "/z/", "/ʃ/", "/ʒ/", "/h/",
    "/m/", "/n/", "/ŋ/", "/l/", "/r/", "/ɹ/", "/w/", "/j/",
    # Retroflex consonants must be in this set too, or consonant-consonant
    # mismatches involving them can never match the retroflex patterns below.
    "/ʈ/", "/ɖ/", "/ɽ/",
    # American English alveolar flap/tap (the "tt"/"dd" in "butter"/"ladder")
    # and the /ts/ affricate cluster — both confirmed present in real detected
    # streams. /ɾ/ is deliberately NOT a mispronunciation signal on its own;
    # see EQUIVALENT_PHONEME_GROUPS below, where it's grouped with /t/ and /d/.
    "/ɾ/", "/ts/",
}

# Every symbol the wav2vec2-espeak CTC model can legitimately emit for English.
# Anything outside this set reaching the expected-vs-detected comparison is
# either multilingual-model noise (see NON_IPA_SALVAGE) or a bug — see
# pronunciation/phoneme_accuracy.py's vocab gate (Fix 2 in the audio-pipeline
# correctness-fix plan).
ENGLISH_IPA_WHITELIST = VOWEL_PHONEMES | CONSONANT_PHONEMES

# (expected, detected) -> (issue_type, severity). One merged table of known
# clarity-affecting consonant substitutions: pattern-specific issue types
# (missing_aspiration, retroflex_consonant) where the shift has a recognized
# articulatory cause, generic consonant_substitution otherwise. Add new known
# clarity-affecting substitutions here — this list intentionally stays
# phoneme-pattern-based, never language/accent-labeled (see mti_analyzer.py's
# module philosophy).
CONSONANT_SUBSTITUTION_PATTERNS = {
    # Devoiced/velar realization of a register feature (dropping the final "g"
    # sound in "-ing"), not a clarity-affecting native-language substitution —
    # kept in this table (so it's graded/labeled consistently everywhere) but
    # with its own issue type and low severity rather than "consonant_substitution".
    ("/ŋ/", "/n/"): ("informal_register", "low"),
    ("/θ/", "/t/"): ("missing_aspiration", "high"),
    ("/θ/", "/s/"): ("missing_aspiration", "medium"),
    ("/θ/", "/f/"): ("consonant_substitution", "medium"),
    ("/ð/", "/d/"): ("missing_aspiration", "high"),
    ("/ð/", "/z/"): ("missing_aspiration", "medium"),
    ("/v/", "/w/"): ("consonant_substitution", "high"),
    ("/w/", "/v/"): ("consonant_substitution", "high"),
    ("/z/", "/dʒ/"): ("consonant_substitution", "medium"),
    ("/z/", "/ʒ/"): ("consonant_substitution", "low"),
    ("/s/", "/z/"): ("consonant_substitution", "low"),
    ("/p/", "/f/"): ("consonant_substitution", "medium"),
    ("/f/", "/p/"): ("consonant_substitution", "medium"),
    ("/ʃ/", "/s/"): ("consonant_substitution", "medium"),
    ("/ʒ/", "/z/"): ("consonant_substitution", "low"),
    ("/r/", "/ɹ/"): ("consonant_substitution", "low"),
    ("/l/", "/r/"): ("consonant_substitution", "medium"),
    ("/r/", "/l/"): ("consonant_substitution", "medium"),
    ("/t/", "/ʈ/"): ("retroflex_consonant", "medium"),
    ("/d/", "/ɖ/"): ("retroflex_consonant", "medium"),
    ("/r/", "/ɽ/"): ("retroflex_consonant", "low"),
}

# (expected, detected) -> severity. Vowel pairs where the detected vowel is a
# lengthened version of the expected one (e.g. "sit" -> "seeet"). Used by
# vowel_patterns.py; kept here so vowel severities live beside consonant ones.
VOWEL_ELONGATION_PAIRS = {
    ("/ɪ/", "/iː/"): "medium",
    ("/ʊ/", "/uː/"): "medium",
    ("/ɛ/", "/ɜː/"): "low",
    ("/ɒ/", "/ɔː/"): "low",
    ("/ʌ/", "/ɜː/"): "low",
}

# Inverse of VOWEL_ELONGATION_PAIRS: detected vowel is a shortened version of expected.
VOWEL_SHORTENING_PAIRS = {(det, exp): sev for (exp, det), sev in VOWEL_ELONGATION_PAIRS.items()}

# Fallback severity for a substitution that isn't in any curated table above.
# Vowel-quality shifts (e.g. a schwa reduction on an unstressed function word
# like "to") are usually accent, not a clarity problem, so an *unknown* vowel
# substitution is treated as low impact. A consonant swap changes word identity
# more, so an unknown consonant substitution stays medium. These are the same
# defaults the MTI vowel/consonant panels use, kept here so the pronunciation
# module grades an error identically — a word can't read one severity in one
# panel and a different one in another.
GENERIC_VOWEL_SEVERITY = "low"
GENERIC_CONSONANT_SEVERITY = "medium"

_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def severity_rank(severity: str) -> int:
    """Numeric rank for a severity label so callers can take a max across errors."""
    return _SEVERITY_RANK.get(severity, 0)


def known_error_severity(expected, detected):
    """
    Severity that the MTI module assigns to a single phoneme substitution, or
    None if it isn't a substitution the MTI panels would list at all (an
    insertion/deletion, or a cross-class vowel<->consonant swap).

    Returns the exact same severity the MTI vowel/consonant panels show — known
    curated patterns get their table severity, everything else gets the generic
    vowel/consonant fallback. The pronunciation module takes the max of this and
    its own count-based severity, which guarantees a word never reads a *lower*
    severity in the pronunciation panel than MTI assigns the same error.
    """
    if expected is None or detected is None:
        return None
    consonant = CONSONANT_SUBSTITUTION_PATTERNS.get((expected, detected))
    if consonant is not None:
        return consonant[1]
    if (expected, detected) in VOWEL_ELONGATION_PAIRS:
        return VOWEL_ELONGATION_PAIRS[(expected, detected)]
    if (expected, detected) in VOWEL_SHORTENING_PAIRS:
        return VOWEL_SHORTENING_PAIRS[(expected, detected)]
    if expected in VOWEL_PHONEMES and detected in VOWEL_PHONEMES:
        return GENERIC_VOWEL_SEVERITY
    if expected in CONSONANT_PHONEMES and detected in CONSONANT_PHONEMES:
        return GENERIC_CONSONANT_SEVERITY
    return None


# ---------------------------------------------------------------------------
# Non-English / multilingual-model noise recovery (Fix 2)
# ---------------------------------------------------------------------------
# The wav2vec2-espeak CTC model is multilingual; it occasionally emits tokens
# outside the English IPA set — empirically confirmed in real detected streams:
# "/a5/", "/y5/" (Mandarin-style tone-tagged units). These can never match any
# expected English phoneme, so every word they land in was previously
# auto-flagged as mispronounced regardless of how the word was actually said.
import re as _re

_TONE_DIGIT_RE = _re.compile(r"\d+$")

# base symbol (after stripping a trailing tone digit) -> valid English IPA
# replacement, for symbols where digit-stripping alone doesn't land on a
# symbol already in ENGLISH_IPA_WHITELIST. Extend as new non-English tokens
# are observed; anything not covered here (and not fixable by stripping) is
# dropped by salvage_phoneme() and the caller marks that region unscorable —
# never compared as-is.
NON_IPA_SALVAGE = {
    "y": "j",  # espeak's own /j/ (as in "yes") is the closest English mapping
}


def salvage_phoneme(raw_symbol: str):
    """
    Normalize a raw CTC-decoded phoneme token (slash-wrapped, e.g. "/y5/") into
    a valid English IPA symbol, or return None if it can't be recovered.

    Order: already valid -> unchanged. Otherwise strip a trailing tone digit
    and check again ("/a5/" -> "/a/", already valid). If the stripped bare
    symbol still isn't valid English IPA, consult NON_IPA_SALVAGE ("/y5/" ->
    strip -> "y" -> not valid alone -> mapped to "/j/"). Returns None if no
    step recovers a valid symbol — the caller should drop the token entirely
    (it must never reach the expected-vs-detected comparison) and mark the
    time region it occupied as unscorable, not "missing".
    """
    if raw_symbol in ENGLISH_IPA_WHITELIST:
        return raw_symbol
    inner = raw_symbol.strip("/")
    stripped = _TONE_DIGIT_RE.sub("", inner)
    if stripped != inner and f"/{stripped}/" in ENGLISH_IPA_WHITELIST:
        return f"/{stripped}/"
    if stripped in NON_IPA_SALVAGE:
        return f"/{NON_IPA_SALVAGE[stripped]}/"
    if inner in NON_IPA_SALVAGE:
        return f"/{NON_IPA_SALVAGE[inner]}/"
    return None


# ---------------------------------------------------------------------------
# Perceptual/acoustic equivalence (Fix 1, Fix 6) — pairs that are NEVER a real
# clarity difference, so alignment should treat them as free matches and MTI
# should never report them as a substitution pattern.
# ---------------------------------------------------------------------------
# NOTE (honest limitation): {ɪ,ə} and {ʊ,u} are only true equivalences in
# UNSTRESSED syllables — but this project's stress detection is itself a
# placeholder (stress_placement.py always reports the expected pattern, no
# real acoustic stress detection yet — see that module's own PLACEHOLDER
# notice), so there is no reliable signal to condition on here. Applied
# unconditionally as a deliberately conservative choice: this may occasionally
# suppress a genuine stressed-syllable vowel error, but that's a much smaller
# harm than the false positives these pairs were causing (schwa reduction on
# ordinary function/unstressed words being reported as a native-language
# pattern). Revisit once real stress detection lands.
EQUIVALENT_PHONEME_GROUPS = [
    frozenset({"/ə/", "/ʌ/"}),
    frozenset({"/ə/", "/ɚ/"}),
    frozenset({"/ɪ/", "/ə/"}),
    frozenset({"/ɾ/", "/t/", "/d/"}),  # American flap — not a real substitution
    frozenset({"/ɑ/", "/ɔ/"}),  # cot-caught merger
    frozenset({"/i/", "/iː/"}),
    frozenset({"/ʊ/", "/u/"}),
]


def are_phonetically_equivalent(expected, detected) -> bool:
    """True if expected/detected are the same sound or a curated equivalent
    pair that must never be scored as a real difference."""
    if expected == detected:
        return True
    if expected is None or detected is None:
        return False
    return any({expected, detected} <= group for group in EQUIVALENT_PHONEME_GROUPS)


def substitution_cost(expected, detected) -> float:
    """
    Edit-distance substitution cost for aligning expected vs. detected phoneme
    sequences (see pronunciation/phoneme_accuracy.py's _align_phonemes) — Fix 1.

    Lower cost = more likely to be the same intended sound, so the alignment
    prefers to resolve it as a direct substitution rather than routing around
    it with extra insertions/deletions (which is what caused cross-word
    misattribution before this fix — e.g. "industry"'s genuinely-missing
    trailing phonemes being walked onto the neighboring word "pharma").

    0.0  = perceptually/acoustically equivalent (see are_phonetically_equivalent) —
           never a real difference.
    0.3  = same broad phonetic class (both vowels, or both consonants) — a
           plausible single-phoneme substitution (whether or not it's one of
           the specifically curated patterns; known_error_severity's generic
           fallback covers any same-class pair), so the alignment should
           prefer treating these as corresponding rather than fragmenting them
           into separate insertion/deletion operations that can misattribute
           real phonemes to a neighboring word (the root cause behind the
           "pharma"/"industry" cross-word bleed this fix addresses).
    1.0  = full cost — a cross vowel/consonant class swap (essentially never a
           real single-phoneme substitution), or expected/detected is None
           (a genuine insertion or deletion).
    """
    if are_phonetically_equivalent(expected, detected):
        return 0.0
    if expected is not None and detected is not None and known_error_severity(expected, detected) is not None:
        return 0.3
    return 1.0


# ---------------------------------------------------------------------------
# Coarticulation (Fix 6) — connected-speech effects at word boundaries that
# are universal across English speakers, never a native-language pattern.
# Requires the phoneme immediately following `expected` in the utterance's
# EXPECTED sequence (cross-word — available because phoneme_accuracy.py builds
# one expected_seq per SENTENCE, not per word, so word boundaries don't break
# this lookup; see that module's docstring on why detection runs per sentence).
# ---------------------------------------------------------------------------
def is_coarticulation_artifact(expected, detected, following_expected_phoneme) -> bool:
    """
    True if (expected -> detected) is explainable by ordinary connected-speech
    coarticulation given what phoneme comes next, rather than a genuine
    native-language pattern. Curated from real observed cases — not
    exhaustive; extend as new legitimate coarticulation cases are confirmed.
    """
    if following_expected_phoneme is None:
        return False
    # Yod-coalescence: word-final /t/ or /d/ + a following /j/ commonly
    # surfaces as an affricate ("about yourself" -> "abou-tcha-self").
    if expected in ("/t/", "/d/") and following_expected_phoneme == "/j/":
        if detected in ("/tʃ/", "/dʒ/"):
            return True
    # Word-final stop weakening/assimilating before another stop or affricate
    # ("good chess" -> final /d/ realized closer to /k/). Broad on purpose:
    # word-final stop release is highly variable in connected speech before
    # any following obstruent, not specific to one place/manner pair.
    _STOPS_AND_AFFRICATES = {"/p/", "/b/", "/t/", "/d/", "/k/", "/ɡ/", "/g/", "/tʃ/", "/dʒ/"}
    if expected in _STOPS_AND_AFFRICATES and following_expected_phoneme in _STOPS_AND_AFFRICATES:
        return True
    return False


# ---------------------------------------------------------------------------
# Function words (Fix 3) — schwa reduction and similar vowel-quality shifts on
# these are ordinary native-speaker connected speech, never MTI. Single source
# of truth: intonation/emphasis.py imports this set rather than keeping its
# own copy (it previously did; moved here so pronunciation's exclusion gates
# and emphasis's content-word heuristic can never drift apart).
# ---------------------------------------------------------------------------
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "is", "am",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "there", "here", "as", "so", "not", "no",
    "yes", "just", "also", "very", "really",
}

# Subset of FUNCTION_WORDS actually prone to unstressed vowel reduction
# (schwa-ing) in ordinary native speech — articles, prepositions, conjunctions,
# auxiliary/copula verbs, pronouns. Deliberately EXCLUDES degree adverbs
# ("very", "really", "just", "also", "not", "yes") that FUNCTION_WORDS also
# contains: those are content-bearing words that don't undergo the same
# reduction, and a real MTI-worthy finding on one of them (e.g. "very" /v/->/w/,
# the original motivating case) must not be suppressed by an over-broad
# stopword list borrowed from a different purpose (emphasis.py's
# is-this-a-content-word heuristic, which reasonably treats degree adverbs as
# non-primary-stress but that's irrelevant to whether their VOWEL is reduced).
# Used only by vowel_patterns.py — consonant substitutions are never exempted
# on function words; that phenomenon (word-final stop weakening, etc.) is
# handled separately by is_coarticulation_artifact.
MTI_VOWEL_EXEMPT_WORDS = FUNCTION_WORDS - {"not", "yes", "just", "also", "very", "really"}

# Minimum number of times a substitution TYPE must recur across a session
# before it's reported as an MTI pattern (Fix 6) — a one-off event goes to
# candidate_patterns instead of the scored output.
MTI_RECURRENCE_MIN = 2

# Tolerance window (seconds) when attributing a detected phoneme to a word by
# timestamp (Fix 1) — CTC phoneme timestamps drift somewhat relative to ASR
# word timestamps, so a phoneme just outside a word's own span may still
# genuinely belong to it.
WORD_PHONEME_TOL_SECONDS = 0.12
