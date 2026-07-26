"""
MTI (Mother Tongue Influence) analysis orchestrator.

Philosophy: this module answers "which native-language speech patterns reduce
interview clarity?" — never "which region/accent does this speaker have?". All
detection here is framed in terms of phoneme-level patterns and their clarity
impact, not speaker classification. Do not add region/accent/native-language
labels to this pipeline's output.

Latency note: vowel_patterns.py and consonant_patterns.py do no audio analysis
of their own — they both reinterpret the same phoneme_errors list already
computed by the pronunciation module's wav2vec2 pass. analyze_mti() accepts that
data pre-computed (phoneme_errors, stress_errors) specifically so the caller
(vad_ui.py) can run wav2vec2 ONCE and feed the same result to both pronunciation
and MTI analysis, instead of paying for a second full acoustic pass.

Each phoneme error is classified in exactly one place: vowel-to-vowel mismatches
in vowel_patterns.py, consonant-to-consonant mismatches in consonant_patterns.py
(the former separate phoneme_substitution.py flagged known substitutions that
are all consonant shifts, so it double-counted; that role was removed).
"""

import logging
import os
import subprocess
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torchaudio

from ..phoneme_patterns import MTI_RECURRENCE_MIN
from .consonant_patterns import analyze_consonant_patterns
from .stress_transfer import analyze_stress_transfer
from .vowel_patterns import analyze_vowel_patterns

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Penalty subtracted from 100 per issue, by severity — a heuristic scoring
# formula (not independently validated against real interview outcomes yet).
SEVERITY_PENALTY = {"low": 2, "medium": 5, "high": 10}

ISSUE_TYPE_LABELS = {
    "phoneme_substitution": "phoneme substitution",
    "vowel_elongation": "elongated vowels",
    "vowel_shortening": "shortened vowels",
    "vowel_substitution": "vowel substitution",
    "missing_aspiration": "missing aspirated sounds",
    "retroflex_consonant": "retroflex consonants",
    "consonant_substitution": "consonant substitution",
    "stress_transfer": "stress transfer",
}


def _load_waveform(audio_path: str, sample_rate: int = SAMPLE_RATE):
    """
    Load audio as a mono waveform at `sample_rate`. Only used in the fallback
    path (when phoneme_errors isn't already provided) — duplicated as a small,
    self-contained helper rather than importing pronunciation_analyzer's private
    loader, same reasoning as that module's own copy of this pattern.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ac", "1",
            "-ar", str(sample_rate),
            "-vn",
            "-loglevel", "error",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode '{audio_path}': {result.stderr.strip()}")
        waveform, sr = torchaudio.load(tmp_path)
        return waveform.squeeze(0), sr
    finally:
        os.remove(tmp_path)


def _apply_recurrence_threshold(
    vowel_patterns: List[Dict], consonant_patterns: List[Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Fix 6: a substitution pattern is only reported as a scored MTI finding if
    the same ISSUE TYPE (vowel_substitution, missing_aspiration, ...) recurs at
    least MTI_RECURRENCE_MIN times across the session — a single one-off slip
    isn't evidence of a recurring native-language pattern. One-offs aren't
    dropped silently; they move to candidate_patterns for visibility.

    Counted by issue TYPE, deliberately, not by exact (expected, detected)
    phoneme pair. An earlier version of this function used the exact pair and
    was far too strict: real speakers repeat a *kind* of substitution across
    different sounds much more often than they repeat one identical phoneme
    swap. Measured on a real session, all 12 detected patterns had a unique
    (expected, detected) pair, so every one of them — including a
    high-severity /w/ -> /v/ — was suppressed, and MTI reported a clean 100
    "low clarity impact" while Pronunciation was simultaneously flagging five
    high-severity words. Counting by issue type keeps the "one-off isn't a
    pattern" intent (a lone vowel substitution is still filtered) without that
    false-negative failure mode, and matches the wording of the original spec.

    Returns (kept_vowel_patterns, kept_consonant_patterns, candidate_patterns).
    """
    tagged = [("vowel", p) for p in vowel_patterns] + [("consonant", p) for p in consonant_patterns]
    issue_counts = Counter(p["issue"] for _, p in tagged)

    kept_vowel, kept_consonant, candidates = [], [], []
    for kind, pattern in tagged:
        if issue_counts[pattern["issue"]] >= MTI_RECURRENCE_MIN:
            (kept_vowel if kind == "vowel" else kept_consonant).append(pattern)
        else:
            candidates.append(pattern)
    return kept_vowel, kept_consonant, candidates


def _classify_clarity_impact(score: int) -> str:
    if score >= 85:
        return "low"
    if score >= 60:
        return "medium"
    return "high"


def _generate_overall_observations(
    overall_mti_score: int,
    clarity_impact: str,
    all_issues: List[Dict],
    top_recurring_issue: Optional[str],
) -> List[str]:
    observations = [f"Overall clarity impact from native-language speech patterns is {clarity_impact}."]

    if not all_issues:
        observations.append("No recurring clarity-affecting patterns detected.")
        return observations

    if top_recurring_issue:
        observations.append(f"Most common pattern: {top_recurring_issue}.")

    high_impact = [
        i for i in all_issues if i.get("severity") == "high" or i.get("clarity_impact") == "high"
    ]
    if high_impact:
        observations.append(f"{len(high_impact)} instance(s) carry high clarity impact.")

    return observations


def analyze_mti(
    audio_path: Optional[str],
    transcript: str,
    words: List[Dict],
    sentences: Optional[List[Dict]] = None,
    phoneme_errors: Optional[List[Dict]] = None,
    stress_errors: Optional[List[Dict]] = None,
) -> Dict:
    """
    Run the full MTI pipeline and combine results into one report.

    Args:
        audio_path: path to interview audio. Only used if phoneme_errors isn't
            already supplied (fallback path — re-runs the wav2vec2 phoneme
            recognizer independently, same as pronunciation_analyzer.py would).
        transcript: full transcript text (kept for interface completeness).
        words: [{"word": str, "start": float, "end": float}, ...]
        sentences: [{"text": str, "start": float, "end": float}, ...], only used
            in the fallback phoneme-detection path (see phoneme_accuracy.py —
            detection runs per sentence, not per word).
        phoneme_errors: pre-computed phoneme_accuracy.analyze_phoneme_accuracy(
            words, sentences, waveform, sample_rate)["errors"], if already
            available from a prior pronunciation analysis pass — pass this to
            avoid a second, redundant wav2vec2 pass over the same audio.
        stress_errors: pre-computed stress_placement.analyze_stress(words)[
            "stress_errors"], likewise.

    Returns:
        {
            "overall_mti_score": int,
            "clarity_impact": str,
            "summary": {...counts...},
            "vowel_patterns": [...],       # recurring only — see candidate_patterns
            "consonant_patterns": [...],   # recurring only — see candidate_patterns
            "stress_transfer": [...],
            "candidate_patterns": [...],
                # One-off substitutions (Fix 6) that didn't meet
                # MTI_RECURRENCE_MIN — not scored, kept for visibility.
            "speech_statistics": {...},
            "patterns_detected": [str, ...],
            "top_recurring_issue": str,
            "clarity_risk_words": [str, ...],
            "overall_observations": [str, ...],
        }
    """
    if phoneme_errors is None:
        logger.info("No pre-computed phoneme errors provided — running phoneme accuracy analysis...")
        from ..pronunciation.phoneme_accuracy import analyze_phoneme_accuracy

        waveform, sample_rate = _load_waveform(audio_path)
        phoneme_errors = analyze_phoneme_accuracy(words, sentences or [], waveform, sample_rate)["errors"]

    if stress_errors is None:
        logger.info("No pre-computed stress errors provided — running stress analysis...")
        from ..pronunciation.stress_placement import analyze_stress

        stress_errors = analyze_stress(words)["stress_errors"]

    logger.info("Analyzing vowel patterns...")
    vowel_patterns = analyze_vowel_patterns(phoneme_errors, words)["vowel_patterns"]

    logger.info("Analyzing consonant patterns...")
    consonant_patterns = analyze_consonant_patterns(phoneme_errors, words)["consonant_patterns"]

    logger.info("Analyzing stress transfer...")
    stress_transfer = analyze_stress_transfer(stress_errors, words)["stress_transfer"]

    # Fix 6: drop one-off substitutions (not evidence of a recurring
    # native-language pattern) from the scored vowel/consonant patterns —
    # moved to candidate_patterns instead of silently discarded.
    vowel_patterns, consonant_patterns, candidate_patterns = _apply_recurrence_threshold(
        vowel_patterns, consonant_patterns
    )

    all_issues = vowel_patterns + consonant_patterns + stress_transfer

    penalty = sum(SEVERITY_PENALTY.get(issue.get("severity", "medium"), 5) for issue in all_issues)
    overall_mti_score = max(0, round(100 - penalty))
    clarity_impact = _classify_clarity_impact(overall_mti_score)

    affected_words = {issue["word"] for issue in all_issues}
    total_words_analyzed = len(words)
    affected_count = len(affected_words)
    affected_percentage = (
        round(100 * affected_count / total_words_analyzed, 1) if total_words_analyzed else 0.0
    )

    issue_type_counts: Dict[str, int] = {}
    for issue in vowel_patterns:
        issue_type_counts[issue["issue"]] = issue_type_counts.get(issue["issue"], 0) + 1
    for issue in consonant_patterns:
        issue_type_counts[issue["issue"]] = issue_type_counts.get(issue["issue"], 0) + 1
    for _ in stress_transfer:
        issue_type_counts["stress_transfer"] = issue_type_counts.get("stress_transfer", 0) + 1

    patterns_detected = [ISSUE_TYPE_LABELS.get(key, key) for key in issue_type_counts]
    top_recurring_key = max(issue_type_counts, key=issue_type_counts.get) if issue_type_counts else None
    top_recurring_issue = ISSUE_TYPE_LABELS.get(top_recurring_key, top_recurring_key)

    clarity_risk_words = list(
        dict.fromkeys(
            issue["word"]
            for issue in all_issues
            if issue.get("severity") == "high" or issue.get("clarity_impact") == "high"
        )
    )

    logger.info(f"MTI analysis complete. Score: {overall_mti_score}, clarity impact: {clarity_impact}")

    result = {
        "overall_mti_score": overall_mti_score,
        "clarity_impact": clarity_impact,
        "summary": {
            "vowel_pattern_issues": len(vowel_patterns),
            "consonant_pattern_issues": len(consonant_patterns),
            "stress_transfer_issues": len(stress_transfer),
        },
        "vowel_patterns": vowel_patterns,
        "consonant_patterns": consonant_patterns,
        "stress_transfer": stress_transfer,
        "candidate_patterns": candidate_patterns,
        "speech_statistics": {
            "total_words_analyzed": total_words_analyzed,
            "affected_words": affected_count,
            "affected_percentage": affected_percentage,
        },
        "patterns_detected": patterns_detected,
        "top_recurring_issue": top_recurring_issue,
        "clarity_risk_words": clarity_risk_words,
        "overall_observations": _generate_overall_observations(
            overall_mti_score, clarity_impact, all_issues, top_recurring_issue
        ),
    }
    # Fix 7: omit the field entirely when there's no recurring issue to name,
    # rather than carry a None that has to be re-checked everywhere it's read.
    if top_recurring_issue is None:
        del result["top_recurring_issue"]
    return result
