"""
LLM rubric-scoring evaluation matrix.

Experiment: stop scoring rubrics with our own deterministic formulas
(pronunciation_score, overall_mti_score, intonation_score, engagement_score,
rhythm_score) and instead have an LLM score each rubric 1-5, judging purely
from raw evidence. Run across multiple LLMs on the same real sessions, so the
scores can be compared against manually-assigned scores to pick which LLM's
judgment is closest — that LLM becomes the fine-tuning target later.

Two hard requirements this module exists to satisfy:
1. The LLM must score independently — it must never see our own computed
   scores, score-derived labels, or narrative verdicts, or it will anchor on
   (or simply parrot) our existing formula instead of forming its own
   judgment. strip_for_judge() removes every such field, keeping only raw,
   objective evidence (counts, timestamps, detected phonemes/words, Hz
   values, severity TAGS on individual items — which are structured
   classification, not a holistic score).
2. Six rubrics, matching the locked schema (docs/coach_data_schema.pdf):
   Fluency, Pronunciation, MTI, Intonation, Rhythm, Engagement.
"""

import copy
import json
import logging
import os
import time
from typing import Dict, List, Optional

from ..llm.anthropic_client import AnthropicLLMClient
from ..llm.config import LLMConfig

logger = logging.getLogger(__name__)

RUBRICS = ["fluency", "pronunciation", "mti", "intonation", "rhythm", "engagement"]

TEACHER_ENV_PREFIX = "ZIQRA_TEACHER_"


# ---------------------------------------------------------------------------
# Stripping — remove every field that is itself a score, a score-derived
# label, or a narrative verdict our own code generated. What remains is raw,
# objective evidence: counts, timestamps, detected phonemes/words, Hz values,
# and structured per-item severity tags (a classification, not a holistic
# judgment of the rubric as a whole).
# ---------------------------------------------------------------------------
def _strip_fluency(fluency: Dict) -> Dict:
    # No aggregate "fluency score" exists in our pipeline today — fillers/
    # pauses/speaking_speed are already raw counts + threshold classifications
    # (e.g. "fast"/"ideal" is a transparent wpm band, not a hidden judgment).
    # Kept as-is.
    return copy.deepcopy(fluency)


def _strip_pronunciation(pronunciation: Dict) -> Dict:
    p = pronunciation
    return {
        "phoneme_accuracy_pct": round(p.get("phoneme_accuracy", 0) * 100),
        "phoneme_errors": p.get("phoneme_errors", []),
        "mispronounced_words": p.get("mispronounced_words", []),
        "skipped_words": p.get("skipped_words", []),
        "clip_edge_errors": p.get("clip_edge_errors", []),
        # stress_accuracy/stress_errors deliberately excluded: stress_placement.py
        # is a hard placeholder (always reports the expected pattern, never a
        # real measurement) — showing it would give the judge false evidence.
        # pronunciation_score / rhythm_score deliberately excluded: those are
        # exactly the numbers this experiment replaces.
    }


def _strip_mti(mti: Dict) -> Dict:
    return {
        "summary": mti.get("summary", {}),
        "vowel_patterns": mti.get("vowel_patterns", []),
        "consonant_patterns": mti.get("consonant_patterns", []),
        "candidate_patterns": mti.get("candidate_patterns", []),
        "speech_statistics": mti.get("speech_statistics", {}),
        "patterns_detected": mti.get("patterns_detected", []),
        "clarity_risk_words": mti.get("clarity_risk_words", []),
        # overall_mti_score / clarity_impact / top_recurring_issue excluded —
        # all three are direct restatements of the hidden score.
    }


def _strip_intonation(intonation: Dict) -> Dict:
    pv = intonation.get("pitch_variation", {})
    ev = intonation.get("energy_variation", {})
    mo = intonation.get("monotonicity", {})
    em = intonation.get("emphasis", {})
    return {
        "pitch": {
            "average_hz": pv.get("average_pitch"),
            "min_hz": pv.get("min_pitch"),
            "max_hz": pv.get("max_pitch"),
            "range_hz": pv.get("pitch_range"),
        },
        "low_energy_sections": ev.get("low_energy_sections", []),
        "monotone_sections": mo.get("monotone_sections", []),
        "under_emphasized_words": em.get("under_emphasized_words", []),
        # pitch_score/energy_score/monotonicity_score/emphasis_score/
        # intonation_score/delivery_label all excluded — every one of them is
        # a hidden per-dimension or composite verdict this experiment replaces.
    }


def _strip_rhythm(pronunciation: Dict) -> Dict:
    # Rhythm is computed inside pronunciation_analyzer.py but is its own rubric
    # per the locked schema — kept as a separate evidence bucket so the prompt
    # structure matches the six rubrics being scored. rhythm_score itself
    # (the number) is excluded; rhythm_issues is a category tag ("uniform word
    # timing"/"erratic pacing"), not a 1-100 score, so it stays as evidence.
    return {"issues": pronunciation.get("rhythm_issues", [])}


def _strip_engagement(engagement: Dict) -> Dict:
    # Engagement is pure synthesis over the other four modules (no acoustic
    # analysis of its own) — once its own score/level/strengths/improvement_areas
    # and every internal sub-score are removed, the only evidence unique to it
    # is which specific windows were flagged problematic. The judge is expected
    # to form its Engagement verdict holistically from ALL the evidence
    # (Fluency + Pronunciation + MTI + Intonation), same as a human coach would
    # — see the system prompt.
    ep = engagement.get("energy_patterns", {})
    pp = engagement.get("pause_patterns", {})
    return {
        "low_energy_segments": ep.get("low_energy_segments", []),
        "disengaging_pauses": pp.get("disengaging_pauses", []),
        "hesitation_clusters": pp.get("hesitation_clusters", []),
    }


def strip_for_judge(analysis: Dict) -> Dict:
    """
    Build the LLM-judge evidence packet from a Level 2 `analysis` dict
    (fluency/pronunciation/mti/intonation/engagement — same shape whether it
    comes straight from audio_analyzer.analyze_audio() or from a stored
    dataset/features/*.json file's "analysis" block).

    Returns evidence grouped by the six rubrics being scored — every field
    still present is raw/objective; every score, score-derived label, and
    narrative verdict our own code generated has been removed. See module
    docstring for why this matters.
    """
    return {
        "fluency": _strip_fluency(analysis["fluency"]),
        "pronunciation": _strip_pronunciation(analysis["pronunciation"]),
        "mti": _strip_mti(analysis["mti"]),
        "intonation": _strip_intonation(analysis["intonation"]),
        "rhythm": _strip_rhythm(analysis["pronunciation"]),
        "engagement": _strip_engagement(analysis["engagement"]),
    }


# ---------------------------------------------------------------------------
# Prompt + judge call
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert speech-language pathologist and interview coach, scoring a candidate's spoken interview response purely from acoustic/linguistic evidence.

You will be given RAW EVIDENCE only — phoneme-level errors, word-level timestamps, detected patterns, pitch/energy statistics, filler and pause counts. There are NO pre-computed scores or verdicts in this evidence; you must form your own independent judgment for each rubric.

Score six rubrics, each on a 1-5 scale (1 = poor, 2 = needs work, 3 = average, 4 = good, 5 = excellent):

1. FLUENCY — filler word frequency, pause/hesitation patterns, speaking pace appropriateness.
2. PRONUNCIATION — phoneme-level accuracy, severity and frequency of mispronounced words.
3. MTI (Mother Tongue Influence) — how much native-language speech patterns (vowel/consonant substitutions) reduce clarity for a listener. Score CLARITY IMPACT, never guess the speaker's origin/accent/region.
4. INTONATION — pitch expressiveness/range, energy consistency, monotone stretches, emphasis on key words.
5. RHYTHM — timing regularity and naturalness of pacing (independent of pronunciation accuracy itself).
6. ENGAGEMENT — your holistic judgment of how an interviewer would perceive this candidate's energy, confidence, and engagement, reasoning over ALL the evidence across every rubric above (this rubric has no unique measurements of its own — it is a synthesis).

Ground every score in the specific evidence given — cite a concrete number, word, or timestamp in your reasoning. Do not invent facts not present in the evidence. Return ONLY the JSON object described, no prose, no markdown fences."""

USER_TEMPLATE = """Evidence for this interview response (transcript duration and rubric-specific raw data):

{evidence_json}

Return ONLY this JSON object (no other text):
{{
  "fluency": {{"score": <1-5 integer>, "reasoning": "<one sentence, cite specific evidence>"}},
  "pronunciation": {{"score": <1-5 integer>, "reasoning": "<one sentence, cite specific evidence>"}},
  "mti": {{"score": <1-5 integer>, "reasoning": "<one sentence, cite specific evidence>"}},
  "intonation": {{"score": <1-5 integer>, "reasoning": "<one sentence, cite specific evidence>"}},
  "rhythm": {{"score": <1-5 integer>, "reasoning": "<one sentence, cite specific evidence>"}},
  "engagement": {{"score": <1-5 integer>, "reasoning": "<one sentence, cite specific evidence>"}}
}}"""


def _validate_judge_output(parsed: Dict) -> bool:
    if not isinstance(parsed, dict):
        return False
    for rubric in RUBRICS:
        entry = parsed.get(rubric)
        if not isinstance(entry, dict):
            return False
        score = entry.get("score")
        if not isinstance(score, (int, float)) or not (1 <= score <= 5):
            return False
    return True


def score_session(evidence: Dict, model: str, api_key: str, base_url: str = "https://opencode.ai/zen/go") -> Dict:
    """Call one LLM to score all six rubrics for one session's stripped evidence."""
    config = LLMConfig(
        api_key=api_key, base_url=base_url, model=model,
        temperature=0.0, max_tokens=2000, timeout_seconds=180.0, max_retries=2,
    )
    client = AnthropicLLMClient(config)
    user_prompt = USER_TEMPLATE.format(evidence_json=json.dumps(evidence, ensure_ascii=False))
    return client.complete_json(
        user_prompt, system=SYSTEM_PROMPT, validate=_validate_judge_output, max_attempts=3
    )


def build_evaluation_matrix(
    sessions: List[Dict], models: List[str], api_key: str, base_url: str = "https://opencode.ai/zen/go"
) -> Dict:
    """
    sessions: [{"session_id": str, "label": str, "analysis": Dict}, ...]
    models: list of model names to run (each session is scored once per model).

    Returns {session_id: {model: {rubric: {"score": int, "reasoning": str}}}}
    plus records any per-call errors under a "_errors" key.
    """
    matrix: Dict[str, Dict] = {}
    errors: List[Dict] = []

    for session in sessions:
        sid, label = session["session_id"], session.get("label", session["session_id"])
        evidence = strip_for_judge(session["analysis"])
        matrix[sid] = {"label": label}
        for model in models:
            logger.info(f"Scoring {label} with {model}...")
            try:
                result = score_session(evidence, model, api_key, base_url)
                matrix[sid][model] = result
            except Exception as e:
                logger.warning(f"  {model} failed on {label}: {e}")
                matrix[sid][model] = None
                errors.append({"session_id": sid, "label": label, "model": model, "error": str(e)})
            time.sleep(1)  # be polite to the API between calls

    matrix["_errors"] = errors
    return matrix
