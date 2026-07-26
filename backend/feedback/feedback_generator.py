"""
LLM feedback generation: turns an evidence_packet.py packet into human-readable
interview coaching via a single prompt-templated call to the configured LLM
(backend/llm/client.py) — no knowledge distillation, no fine-tuning.

Why prompt-only: the feature extractors already do all the measurement (VAD,
STT, phoneme/pitch/energy analysis). This layer's only job is *narration* —
turning already-correct, already-consistent structured data into readable
coaching prose. That's a well-suited task for a strong general model with a
good prompt; it doesn't need a model trained specifically for it. A
fine-tuned/distilled model may make sense for later phases (Language/Interview
Intelligence) where the LLM performs genuine judgment calls, not narration —
that's a decision for when those phases are built, on their own evidence.

Grounding is enforced two ways: (1) the system prompt instructs the model to
use ONLY the evidence packet and pass scores through unchanged, (2) a soft
post-hoc check logs a warning (does not block display) if the response
mentions a percentage/score not traceable to the packet.
"""

import json
import logging
import os
import re
from typing import Dict

from backend.llm.client import LLMClient, LLMConfigError, LLMRequestError

logger = logging.getLogger(__name__)

_SAMPLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "feedback_samples", "samples.jsonl"
)

REQUIRED_KEYS = {"overall_assessment", "key_strengths", "priority_improvements", "practice_tips"}

SYSTEM_PROMPT = """You are an expert interview coach giving feedback on a candidate's spoken interview answer.

You will receive a JSON "evidence packet" containing scores and timestamped observations already computed by \
audio analysis (pronunciation, mother-tongue-influence/clarity, intonation, engagement, fluency). Your job is to \
turn this structured evidence into clear, encouraging, human coaching feedback.

STRICT RULES — do not break these:
1. Use ONLY information present in the evidence packet. Do not invent scores, percentages, timestamps, or claims \
not supported by the packet.
2. Any score or percentage you mention must be the exact value from the packet, or a direct quote of it — never a \
new or re-derived number.
3. Never label or speculate about the speaker's region, accent, nationality, or native language. The "mti" section \
is about phoneme-level clarity patterns only (e.g. "the /th/ sound was substituted with /t/") — never turn this \
into "this sounds like an X accent" or similar. Frame everything in terms of interview clarity, not identity.
4. Be specific: reference the actual words, timestamps, and observations from the packet rather than generic advice.
5. Be constructive and encouraging in tone, like a good coach — not just a list of flaws.

Respond with ONLY a single JSON object (no markdown fences, no prose before or after) matching exactly this schema:
{
  "overall_assessment": "2-4 sentence overall summary of the delivery",
  "key_strengths": ["short strength statement", ...],
  "priority_improvements": [
    {"issue": "short issue name", "evidence": "the specific score/word/timestamp from the packet supporting this", "suggestion": "concrete, actionable suggestion"},
    ...
  ],
  "practice_tips": ["short, actionable practice tip", ...]
}"""


def _validate_schema(parsed) -> bool:
    return isinstance(parsed, dict) and REQUIRED_KEYS.issubset(parsed.keys())


def _collect_known_numbers(evidence_packet: Dict) -> set:
    """Every score/percentage value actually present in the packet, for the
    soft grounding check below."""
    numbers = set()
    for section in ("pronunciation", "mti", "intonation", "engagement"):
        for key, val in evidence_packet.get(section, {}).items():
            if isinstance(val, (int, float)):
                numbers.add(round(val))
    return numbers


def _warn_if_ungrounded_numbers(feedback_text: str, evidence_packet: Dict) -> None:
    """
    Soft check only — logs a warning, never blocks display. Looks for
    "NN%" or "NN/100" style mentions and flags any not traceable to the
    packet's own score fields. Heuristic: small/common numbers (0-10) are
    ignored since they're often list counts, not fabricated scores.
    """
    known = _collect_known_numbers(evidence_packet)
    mentioned = {int(n) for n in re.findall(r"\b(\d{1,3})(?:%|/100)", feedback_text)}
    ungrounded = {n for n in mentioned if n > 10 and n not in known}
    if ungrounded:
        logger.warning(
            f"LLM feedback mentions score-like number(s) not found in the evidence packet: {ungrounded}. "
            "Possible ungrounded/hallucinated figures — review before trusting."
        )


def _log_sample(evidence_packet: Dict, feedback: Dict) -> None:
    """Append a (packet, feedback) pair for manual prompt-quality review across
    iterations. Not a training dataset — see this module's docstring on why no
    distillation happens here."""
    try:
        os.makedirs(os.path.dirname(_SAMPLES_PATH), exist_ok=True)
        with open(_SAMPLES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"evidence_packet": evidence_packet, "feedback": feedback}, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning(f"Could not write feedback sample log: {e}")


def generate_feedback(evidence_packet: Dict) -> Dict:
    """
    Generate structured coaching feedback from an evidence packet via a single
    templated LLM call. JSON parsing/retry-on-invalid-JSON is handled generically
    by LLMClient.complete_json() — this function only supplies the feedback-
    specific prompt and schema check.

    Always returns a dict — never raises. On any failure (missing config,
    request failure, invalid JSON after retrying) the dict has an "error" key
    instead of the schema fields, so callers (the UI) can handle every outcome
    the same way: check for "error".
    """
    user_prompt = "Evidence packet:\n\n" + json.dumps(evidence_packet, indent=2, ensure_ascii=False)

    try:
        client = LLMClient()
        feedback = client.complete_json(user_prompt, system=SYSTEM_PROMPT, validate=_validate_schema)
    except LLMConfigError as e:
        return {"error": str(e)}
    except LLMRequestError as e:
        return {"error": str(e)}

    _warn_if_ungrounded_numbers(json.dumps(feedback, ensure_ascii=False), evidence_packet)
    _log_sample(evidence_packet, feedback)
    return feedback
