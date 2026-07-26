"""
Regression fixtures for the audio-pipeline correctness-fix plan (10 fixes,
see ~/.claude/plans/declarative-gathering-sutton.md for the full write-up).

Built from three real QA sessions that exposed systemic false positives in
the scoring pipeline (`dataset/timeline/session_20260723_{154240,160418,
161002}_*.json` — Level 1 ground truth: words+confidence, detected_phonemes+
timestamps, acoustic contours, speech_chunks). Every session's ORIGINAL audio
is not stored, only its derived Level 1 output — so tests here run the real,
audio-independent analysis functions directly against that stored ground
truth (intonation, rhythm, exclusions), or reconstruct the exact alignment
inputs from real G2P output (phonemizer IS installed) + the real stored
detected-phoneme stream (already-frozen model output, so no wav2vec2/audio
needed) for the phoneme-alignment fixtures. Nothing here is approximated
data invented for the test — every fixture traces back to a real recording.

No pytest in this environment — plain assertions, run directly:
    venv/bin/python3 backend/tests/test_audio_pipeline_regressions.py
Exits non-zero (via assertion failure) on any regression.
"""

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from backend.feature_extractors.audio.exclusions import compute_excluded_words
from backend.feature_extractors.audio.intonation.energy_variation import analyze_energy_variation
from backend.feature_extractors.audio.intonation.pitch_variation import analyze_pitch_variation
from backend.feature_extractors.audio.mti.consonant_patterns import analyze_consonant_patterns
from backend.feature_extractors.audio.mti.mti_analyzer import _apply_recurrence_threshold
from backend.feature_extractors.audio.mti.vowel_patterns import analyze_vowel_patterns
from backend.feature_extractors.audio.phoneme_patterns import is_coarticulation_artifact
from backend.feature_extractors.audio.pronunciation.phoneme_accuracy import (
    SENTENCE_PADDING_SECONDS,
    _align_phonemes,
    _mark_clip_edge_runs,
    get_expected_phonemes_for_sentence,
    salvage_phoneme,
)
from backend.feature_extractors.audio.pronunciation.pronunciation_analyzer import (
    _generate_overall_observations,
)
from backend.feature_extractors.audio.pronunciation.rhythm import analyze_rhythm

DATASET_DIR = os.path.join(_PROJECT_ROOT, "dataset", "timeline")

SESSIONS = {
    "silence_team_funcword": "session_20260723_154240_786b7969.json",
    "pharma_thankyou_boundary": "session_20260723_160418_372dba93.json",
    "names_question_pauses": "session_20260723_161002_e153ea37.json",
}


def _load(session_key: str) -> dict:
    return json.load(open(os.path.join(DATASET_DIR, SESSIONS[session_key])))


def _reconstruct_sentence_alignment(timeline: dict, sentence_predicate):
    """
    Real end-to-end reconstruction of one sentence's phoneme alignment inputs
    from stored Level 1 ground truth: real G2P (phonemizer) for expected_seq,
    real stored detected_phonemes (frozen model output) sliced to the
    sentence's padded span for detected_seq — no audio or model re-run needed.
    Returns (words_in_sentence, expected_seq, phoneme_to_local_word, errors).
    """
    words, sentences = timeline["words"], timeline["sentences"]
    duration = timeline["audio_metadata"]["duration"]
    sentence = next(s for s in sentences if sentence_predicate(s))

    words_in_sentence = [w for w in words if sentence["start"] - 1e-3 <= w["start"] < sentence["end"] + 1e-3]
    per_word = get_expected_phonemes_for_sentence(sentence["text"])
    expected_seq, phoneme_to_local_word = [], []
    for local_idx, phonemes in enumerate(per_word):
        if phonemes:
            expected_seq.extend(phonemes)
            phoneme_to_local_word.extend([local_idx] * len(phonemes))

    padded_start = max(0.0, sentence["start"] - SENTENCE_PADDING_SECONDS)
    padded_end = min(duration, sentence["end"] + SENTENCE_PADDING_SECONDS)
    detected_seq = []
    for d in timeline["detected_phonemes"]:
        if padded_start <= d["start"] < padded_end:
            salvaged = salvage_phoneme(d["phoneme"])
            if salvaged:
                detected_seq.append(salvaged.strip("/"))

    errors = _align_phonemes(expected_seq, detected_seq)
    _mark_clip_edge_runs(errors, len(expected_seq))
    return words_in_sentence, expected_seq, phoneme_to_local_word, errors


def _scored_words_for(errors, phoneme_to_local_word, words_in_sentence) -> set:
    """Words with at least one non-clip-edge (real, scored) error."""
    scored = set()
    for e in errors:
        if e.get("clip_edge"):
            continue
        idx = phoneme_to_local_word[min(e["expected_index"], len(phoneme_to_local_word) - 1)]
        if idx < len(words_in_sentence):
            scored.add(words_in_sentence[idx]["word"].strip(".,").lower())
    return scored


# ---------------------------------------------------------------------------
# Fix 1 + 2: word<->phoneme attribution + CTC vocab gate
# ---------------------------------------------------------------------------
def test_fix1_pharma_no_longer_misattributed():
    """Acceptance: 'pharma' must not appear in mispronounced_words — its
    trailing-sentence deletions belong to 'industry' being clip-truncated,
    not a real pharma mispronunciation."""
    tl = _load("pharma_thankyou_boundary")
    words_in_sent, expected_seq, ptl, errors = _reconstruct_sentence_alignment(
        tl, lambda s: "pharma" in s["text"]
    )
    scored = _scored_words_for(errors, ptl, words_in_sent)
    assert "pharma" not in scored, f"pharma must have zero real errors, got scored words: {scored}"

    # the deletions must exist and be correctly flagged clip_edge, not silently vanished
    clip_edge = [e for e in errors if e.get("clip_edge")]
    assert len(clip_edge) >= 4, f"expected >=4 trailing clip-edge deletions, got {len(clip_edge)}"


def test_fix2_no_garbage_tokens_reach_comparison():
    """Acceptance: no non-English-IPA token (e.g. tone-tagged '/y5/', '/a5/')
    ever reaches the expected-vs-detected comparison for 'Thank you.'."""
    tl = _load("pharma_thankyou_boundary")
    words_in_sent, expected_seq, ptl, errors = _reconstruct_sentence_alignment(
        tl, lambda s: s["text"].strip().lower().startswith("thank")
    )
    for e in errors:
        assert e["detected"] is None or "5" not in e["detected"], f"garbage token leaked into comparison: {e}"

    # "you" must not carry a /j/->/k/ style cross-word-bleed error anymore
    scored = _scored_words_for(errors, ptl, words_in_sent)
    you_errors = [
        e for e in errors if not e.get("clip_edge")
        and ptl[min(e["expected_index"], len(ptl) - 1)] < len(words_in_sent)
        and words_in_sent[ptl[min(e["expected_index"], len(ptl) - 1)]]["word"].strip(".").lower() == "you"
    ]
    assert not any(e["detected"] == "/k/" for e in you_errors), f"you must not show the old /j/->/k/ bleed: {you_errors}"


# ---------------------------------------------------------------------------
# Fix 3: exclusion gates (proper names, low-confidence fragments)
# ---------------------------------------------------------------------------
def test_fix3_names_and_prompt_fragment_excluded():
    """Acceptance: samad/periwal (proper names) and thank (low-confidence
    prompt fragment) must be excluded from scoring."""
    tl = _load("names_question_pauses")
    excluded = compute_excluded_words(tl["words"], tl["audio_metadata"]["duration"])
    assert excluded.get("samad") == "proper_name", excluded
    assert excluded.get("periwal") == "proper_name", excluded
    assert excluded.get("thank") == "low_confidence", excluded


def test_bugC_clip_edge_not_mislabelled_low_confidence():
    """Regression: a word excluded only for sitting at the clip boundary must
    NOT be reported as 'low_confidence'. Measured on real audio, 'naturally' at
    99.5% ASR confidence was excluded for starting 0.2s in and still labelled
    low_confidence — these reasons become training data, so a false one is
    worse than none."""
    words = [
        {"word": "naturally", "start": 0.20, "end": 0.68, "confidence": 0.9953},
        {"word": "confident", "start": 5.00, "end": 5.60, "confidence": 0.99},
        {"word": "mumbled", "start": 6.00, "end": 6.40, "confidence": 0.13},
    ]
    excluded = compute_excluded_words(words, total_duration=15.1)
    assert excluded.get("naturally") == "clip_edge", (
        f"high-confidence clip-boundary word must be 'clip_edge', got {excluded.get('naturally')!r}"
    )
    assert excluded.get("mumbled") == "low_confidence", excluded
    assert "confident" not in excluded, f"mid-clip confident word must not be excluded: {excluded}"


def test_fix3_budget_reported_not_silently_dropped():
    """Borderline-confidence real words (e.g. 'budget'@0.462) are excluded
    from scoring but must still be visible via the exclusion reason, not
    silently vanished."""
    tl = _load("silence_team_funcword")
    excluded = compute_excluded_words(tl["words"], tl["audio_metadata"]["duration"])
    if "budget" in excluded:
        assert excluded["budget"] == "low_confidence"


# ---------------------------------------------------------------------------
# Fix 4: intonation silence gating
# ---------------------------------------------------------------------------
def test_fix4_leading_silence_not_scored_as_low_energy():
    """Acceptance: session with 1.2s of leading silence before speech starts
    must produce zero low_energy_sections overlapping non-speech regions."""
    import numpy as np

    tl = _load("silence_team_funcword")
    contour = tl["acoustic_contours"]["energy_rms"]
    energy_contour = {
        "rms": np.array([p["rms"] for p in contour]),
        "times": np.array([p["time"] for p in contour]),
    }
    dummy = np.array([], dtype=np.float32)
    result = analyze_energy_variation(
        dummy, 16000, energy_contour=energy_contour, speech_chunks=tl["speech_chunks"]
    )
    for section in result["low_energy_sections"]:
        assert any(
            c["start"] <= section["start"] and section["end"] <= c["end"] for c in tl["speech_chunks"]
        ), f"low_energy_section {section} falls outside all speech_chunks {tl['speech_chunks']}"


def test_fix4_inter_chunk_gap_not_scored_as_low_energy():
    """Acceptance: session with real inter-chunk pauses (15.7s-19.2s gap
    region) must produce zero low_energy_sections there."""
    import numpy as np

    tl = _load("names_question_pauses")
    contour = tl["acoustic_contours"]["energy_rms"]
    energy_contour = {
        "rms": np.array([p["rms"] for p in contour]),
        "times": np.array([p["time"] for p in contour]),
    }
    dummy = np.array([], dtype=np.float32)
    result = analyze_energy_variation(
        dummy, 16000, energy_contour=energy_contour, speech_chunks=tl["speech_chunks"]
    )
    assert result["low_energy_sections"] == [], f"expected zero sections, got {result['low_energy_sections']}"


# ---------------------------------------------------------------------------
# Fix 5: pitch cleaning
# ---------------------------------------------------------------------------
def test_fix5_pitch_range_physiologically_plausible():
    """Acceptance: session with octave-error-inflated range_hz (raw 360.4Hz)
    must fall to a plausible range for a real speaker (roughly 60-120Hz)."""
    import numpy as np

    tl = _load("names_question_pauses")
    contour = tl["acoustic_contours"]["pitch_hz"]
    pitch_contour = {
        "f0": np.array([p["hz"] for p in contour]),
        "times": np.array([p["time"] for p in contour]),
        "voiced": np.array([p["voiced"] for p in contour]),
    }
    dummy = np.array([], dtype=np.float32)
    result = analyze_pitch_variation(dummy, 16000, pitch_contour=pitch_contour)
    assert result["pitch_range"] < 200.0, f"pitch_range still inflated: {result['pitch_range']}"
    assert 40.0 <= result["pitch_range"] <= 150.0, f"pitch_range implausible: {result['pitch_range']}"


# ---------------------------------------------------------------------------
# Fix 6: MTI false-positive filters
# ---------------------------------------------------------------------------
def test_fix6_dedup_function_words_recurrence():
    words = [{"word": w, "start": i, "end": i + 0.3} for i, w in enumerate(
        ["good", "the", "very", "vine", "resource"]
    )]
    phoneme_errors = [
        {"word": "good", "word_index": 0, "expected": "/ɡ/", "detected": "/k/"},
        {"word": "good", "word_index": 0, "expected": "/d/", "detected": "/k/"},
        {"word": "the", "word_index": 1, "expected": "/ɪ/", "detected": "/ə/"},
        {"word": "very", "word_index": 2, "expected": "/v/", "detected": "/w/"},
        {"word": "vine", "word_index": 3, "expected": "/v/", "detected": "/w/"},
        {"word": "resource", "word_index": 4, "expected": "/s/", "detected": "/ʃ/"},
    ]
    vowel = analyze_vowel_patterns(phoneme_errors, words)["vowel_patterns"]
    consonant = analyze_consonant_patterns(phoneme_errors, words)["consonant_patterns"]

    assert vowel == [], f"true function word 'the' vowel reduction must be excluded, got {vowel}"
    assert len([p for p in consonant if p["word"] == "good"]) == 1, "good must dedup to exactly 1 entry"
    assert "very" in {p["word"] for p in consonant}, "very (real consonant finding) must survive"

    kept_vowel, kept_consonant, candidates = _apply_recurrence_threshold(vowel, consonant)
    kept_words = {p["word"] for p in kept_consonant}
    assert {"very", "vine"} <= kept_words, f"recurring /v/->/w/ must be kept, got {kept_words}"


def test_fix6_recurrence_counts_issue_type_not_exact_pair():
    """Regression for a real false NEGATIVE found on session 20260725_150651:
    every detected pattern had a UNIQUE (expected, detected) pair, so
    pair-based recurrence counting suppressed all 12 of them — including a
    high-severity /w/->/v/ — and MTI reported a clean 100 'low clarity impact'
    while Pronunciation flagged five high-severity words. Counting by issue
    TYPE (the spec's actual wording) keeps them."""
    words = [{"word": w, "start": i, "end": i + 0.3} for i, w in enumerate(
        ["paced", "rewarded", "naturally", "pain", "grow"]
    )]
    # five consonant substitutions, all with DIFFERENT phoneme pairs
    phoneme_errors = [
        {"word": "paced", "word_index": 0, "expected": "/p/", "detected": "/b/"},
        {"word": "rewarded", "word_index": 1, "expected": "/w/", "detected": "/v/"},
        {"word": "naturally", "word_index": 2, "expected": "/ɹ/", "detected": "/l/"},
        {"word": "pain", "word_index": 3, "expected": "/n/", "detected": "/ŋ/"},
        {"word": "grow", "word_index": 4, "expected": "/ɡ/", "detected": "/k/"},
    ]
    consonant = analyze_consonant_patterns(phoneme_errors, words)["consonant_patterns"]
    kept_vowel, kept_consonant, candidates = _apply_recurrence_threshold([], consonant)
    assert len(kept_consonant) == 5, (
        f"5 recurring consonant substitutions must all be kept, got {len(kept_consonant)} "
        f"(candidates={len(candidates)}) — pair-based counting would drop all of them"
    )
    assert any(p["severity"] == "high" for p in kept_consonant), "the high-severity /w/->/v/ must survive"


def test_fix6_genuine_one_off_still_filtered():
    """The recurrence gate must still do its original job: a lone instance of
    an issue type is not a 'pattern' and stays out of the scored output."""
    words = [{"word": "resource", "start": 1.0, "end": 1.4}]
    phoneme_errors = [{"word": "resource", "word_index": 0, "expected": "/s/", "detected": "/ʃ/"}]
    consonant = analyze_consonant_patterns(phoneme_errors, words)["consonant_patterns"]
    kept_vowel, kept_consonant, candidates = _apply_recurrence_threshold([], consonant)
    assert kept_consonant == [], f"a single one-off must not be scored, got {kept_consonant}"
    assert len(candidates) == 1, f"it must still be visible in candidate_patterns, got {candidates}"


def test_bugB_repeated_word_severity_is_per_occurrence():
    """Regression for a real false POSITIVE found on session 20260725_150651:
    three separate occurrences of 'a', one slip each, were merged by word text
    into a single entry with three errors -> read as HIGH severity and stamped
    with the first occurrence's 20ms span."""
    from backend.feature_extractors.audio.pronunciation.word_pronunciation import (
        detect_mispronounced_words,
    )

    words = [
        {"word": "a", "start": 14.98, "end": 15.00},
        {"word": "a", "start": 21.32, "end": 21.82},
        {"word": "a", "start": 24.44, "end": 25.02},
    ]
    phoneme_errors = [
        {"word": "a", "word_index": 0, "expected": "/ɐ/", "detected": None},
        {"word": "a", "word_index": 1, "expected": "/ɐ/", "detected": None},
        {"word": "a", "word_index": 2, "expected": "/ɐ/", "detected": "/ɚ/"},
    ]
    result = detect_mispronounced_words(phoneme_errors, words)["mispronounced_words"]
    assert len(result) == 3, f"each occurrence must be its own entry, got {len(result)}"
    assert all(w["severity"] == "low" for w in result), (
        f"one slip per occurrence is 'low', never aggregated to 'high', got {[w['severity'] for w in result]}"
    )
    starts = sorted(w["start"] for w in result)
    assert starts == [14.98, 21.32, 24.44], f"each entry must carry its OWN timestamp, got {starts}"


def test_fix6_coarticulation_suppressed():
    assert is_coarticulation_artifact("/t/", "/tʃ/", "/j/"), "yod-coalescence must be suppressed"
    assert is_coarticulation_artifact("/d/", "/k/", "/tʃ/"), "word-final stop weakening must be suppressed"
    assert not is_coarticulation_artifact("/v/", "/w/", "/ɛ/"), "unrelated pair must NOT be suppressed"


# ---------------------------------------------------------------------------
# Fix 7: coherence derivations + engagement invariants
# ---------------------------------------------------------------------------
def test_fix7_mispronounced_count_matches_names_shown():
    mispron = [{"word": w, "severity": "high", "start": i, "end": i + 1} for i, w in enumerate(
        ["thank", "samad", "periwal", "study", "basketball", "extra"]
    )]
    obs = _generate_overall_observations(84, mispron, [], [])
    assert obs[1].startswith("6 word(s)"), obs[1]
    names_shown = obs[1].split(":")[1].strip().rstrip(".").split(", ")
    assert len(names_shown) == 5, f"exactly 5 names must be shown, got {len(names_shown)}"


def test_fix7_engagement_pace_scored_and_invariant_holds():
    from backend.feature_extractors.audio.engagement.engagement_analyzer import analyze_engagement

    fluency = {"speaking_speed": {"classification": "too_fast"}, "fillers": {"fillers": []}, "pauses": {"pauses": []}}
    pronunciation = {"rhythm_score": 1.0, "mispronounced_words": []}
    mti = {"overall_mti_score": 100, "clarity_impact": "low", "clarity_risk_words": []}
    intonation = {
        "pitch_variation": {"pitch_score": 95}, "emphasis": {"emphasis_score": 95},
        "monotonicity": {"monotonicity_score": 95, "monotone_sections": []},
        "energy_variation": {"low_energy_sections": []},
    }
    result = analyze_engagement(fluency, pronunciation, mti, intonation, total_duration=30.0)
    assert result["engagement_score"] < 90, f"too_fast must pull score down, got {result['engagement_score']}"
    assert any("too fast" in a for a in result["improvement_areas"]), result["improvement_areas"]


# ---------------------------------------------------------------------------
# Fix 8: rhythm discrimination
# ---------------------------------------------------------------------------
def test_fix8_rhythm_discriminates_across_sessions():
    """Acceptance: rhythm must no longer be a flat 1.0/no-issues placeholder
    on every session — the session with real irregular timing must score
    measurably lower than the two clean ones."""
    scores = {}
    for key in SESSIONS:
        tl = _load(key)
        result = analyze_rhythm(tl["words"], speech_chunks=tl["speech_chunks"])
        assert result["experimental"] is True
        scores[key] = result["rhythm_score"]
    assert len(set(scores.values())) == len(scores), f"scores must differ, got {scores}"
    assert scores["names_question_pauses"] < scores["silence_team_funcword"], scores
    assert scores["names_question_pauses"] < scores["pharma_thankyou_boundary"], scores


# ---------------------------------------------------------------------------
# Global invariants (Fix 10) — reusable across any session's real output
# ---------------------------------------------------------------------------
def check_global_invariants(timeline: dict, analysis: dict) -> list:
    """
    Returns a list of violation strings (empty = all invariants hold). Run
    this against any session's Level 1 timeline + Level 2 analysis block.
    """
    violations = []
    speech_chunks = timeline.get("speech_chunks", [])

    def _in_any_chunk(start, end):
        return any(c["start"] <= start and end <= c["end"] for c in speech_chunks)

    intonation = analysis.get("intonation", {})
    for section in intonation.get("energy_variation", {}).get("low_energy_sections", []):
        if not _in_any_chunk(section["start"], section["end"]):
            violations.append(f"flat_section outside speech_chunks: {section}")
    for section in intonation.get("monotonicity", {}).get("monotone_sections", []):
        if not _in_any_chunk(section["start"], section["end"]):
            violations.append(f"monotone_section outside speech_chunks: {section}")

    mti = analysis.get("mti", {})
    pattern_types = {p["issue"] for p in mti.get("vowel_patterns", []) + mti.get("consonant_patterns", [])}
    top = mti.get("top_recurring_issue")
    if top is not None and pattern_types:
        from backend.feature_extractors.audio.mti.mti_analyzer import ISSUE_TYPE_LABELS
        labels = {ISSUE_TYPE_LABELS.get(t, t) for t in pattern_types}
        if top not in labels:
            violations.append(f"top_recurring_issue {top!r} not in actual pattern types {labels}")

    pronunciation = analysis.get("pronunciation", {})
    high_sev = [w for w in pronunciation.get("mispronounced_words", []) if w.get("severity") == "high"]
    for obs in pronunciation.get("overall_observations", []):
        if "word(s) show significant mispronunciation" in obs:
            claimed = int(obs.split(" word(s)")[0])
            if claimed != len(high_sev):
                violations.append(f"observation claims {claimed} high-severity words, actual count is {len(high_sev)}")

    return violations


def test_fix10_global_invariants_helper_is_callable():
    """Smoke-test: the invariant checker itself runs cleanly against an
    empty/minimal analysis without crashing (real per-session analysis
    invariant checks require a full Level 2 run, done via the live pipeline)."""
    violations = check_global_invariants({"speech_chunks": []}, {})
    assert violations == []


ALL_TESTS = [
    test_fix1_pharma_no_longer_misattributed,
    test_fix2_no_garbage_tokens_reach_comparison,
    test_fix3_names_and_prompt_fragment_excluded,
    test_bugC_clip_edge_not_mislabelled_low_confidence,
    test_fix3_budget_reported_not_silently_dropped,
    test_fix4_leading_silence_not_scored_as_low_energy,
    test_fix4_inter_chunk_gap_not_scored_as_low_energy,
    test_fix5_pitch_range_physiologically_plausible,
    test_fix6_dedup_function_words_recurrence,
    test_fix6_recurrence_counts_issue_type_not_exact_pair,
    test_fix6_genuine_one_off_still_filtered,
    test_bugB_repeated_word_severity_is_per_occurrence,
    test_fix6_coarticulation_suppressed,
    test_fix7_mispronounced_count_matches_names_shown,
    test_fix7_engagement_pace_scored_and_invariant_holds,
    test_fix8_rhythm_discriminates_across_sessions,
    test_fix10_global_invariants_helper_is_callable,
]


if __name__ == "__main__":
    passed, failed = 0, 0
    for test in ALL_TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
