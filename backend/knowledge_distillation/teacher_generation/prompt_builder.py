"""
Prompt Builder — Coach Packet -> Prompt (Part 3 of teacher_generation).

Turns an Articulation or Delivery coach packet (Level 1 Timeline + Level 2
Analytics, already segregated per coach by
backend.feedback.coach_packets.build_coach_packets()) into the exact prompt
text sent to that coach's teacher model:

    Articulation packet -> build_articulation_prompt() -> MiniMax M3
    Delivery packet     -> build_delivery_prompt()     -> MiMo-v2.5

This module builds text only. It takes a plain packet dict + session_id and
returns a plain prompt string — nothing else. It does not call a teacher
model (that's provider.py, Part 2) and does not build coach packets (that's
packet_builder.py); it imports neither of them, nor schemas.py, so it stays
usable and testable independently of both (enforced by a module-boundary
test). The returned string is ready to pass straight to
TeacherProvider.generate_articulation() / .generate_delivery().

Every prompt instructs the teacher to produce, in this order:

    1. scores          -- rubric scores (1-5), one per rubric this coach
                           owns, mirroring the existing LLM-judge convention
                           in backend/distillation/llm_eval_matrix.py.
    2. coach_output     -- the team-locked Articulation/Delivery Coach JSON
                           (docs/coach_output_schema.md).
    3. reasoning_trace  -- a structured explanation of how the supplied
                           evidence led to each score and finding above.

Rubric split follows backend/feedback/coach_packets.py's segmental/
suprasegmental seam: Articulation scores {pronunciation, mti}; Delivery
scores {fluency, intonation, engagement}. The Delivery packet also carries
rhythm evidence (rhythm rides with Delivery per that module's docstring),
but per this task's spec rhythm is not scored as its own rubric here — its
evidence still feeds the "rhythm" detailed_findings category the Delivery
Coach schema already defines.
"""

import json
from typing import Any, Dict, Sequence

ARTICULATION_TEACHER = "MiniMax M3"
DELIVERY_TEACHER = "MiMo-v2.5"

ARTICULATION_RUBRICS = ("pronunciation", "mti")
DELIVERY_RUBRICS = ("fluency", "intonation", "engagement")

ARTICULATION_SCHEMA = """{
  "session_id":            str,
  "coach":                 "articulation",
  "overall_assessment":    {"level": str, "summary": str},
  "strengths":              [{"title": str, "description": str}],
  "priority_improvements":  [{"priority": int, "issue": str, "impact": str, "why_it_matters": str}],
  "detailed_findings":      [{"word": str, "timestamp": {"start": float, "end": float}, "issue": str,
                              "expected": str, "detected": str, "severity": str, "impact": str, "explanation": str}],
  "recurring_patterns":     [{"pattern": str, "frequency": str, "affected_words": [str], "overall_impact": str}],
  "practice_plan":          [{"focus": str, "exercise": str}],
  "review_timeline":        [{"time": str, "event": str}],
  "next_session_focus":     [str]
}"""

DELIVERY_SCHEMA = """{
  "session_id":            str,
  "coach":                 "delivery",
  "overall_assessment":    {"delivery_level": str, "summary": str},
  "interviewer_impression": {"perceived_confidence": str, "perceived_engagement": str,
                             "communication_flow": str, "professional_presence": str, "overall_impact": str},
  "strengths":              [{"title": str, "description": str}],
  "priority_improvements":  [{"priority": int, "issue": str, "impact": str, "why_it_matters": str}],
  "detailed_findings":      [{"category": str, "severity": str, "observation": str, "impact": str,
                              "evidence": "shape varies by category -- see below"}],
  "timeline_review":        [{"time": str, "event": str}],
  "behavioral_patterns":    [{"pattern": str, "frequency": str, "effect": str}],
  "practice_plan":          [{"focus": str, "exercise": str}],
  "coach_priority":         {"fix_first": str, "reason": str, "estimated_delivery_gain": str},
  "next_session_focus":     [str]
}"""

DELIVERY_EVIDENCE_SHAPES = """detailed_findings[].evidence shape by category:
  pace    -> {"words_per_minute": int, "classification": str}
  fillers -> {"examples": [str]}
  pauses  -> {"dead_air": int, "long_pauses": int, "hesitations": int}
  energy  -> {"timestamps": [{"start": float, "end": float}]}
  rhythm  -> {"timestamps": [{"start": float, "end": float}]}"""

_ROLE_PREAMBLE = """You are {teacher_name}, acting as the {role_title} for a spoken interview-practice platform. You evaluate ONE candidate's spoken response using only the deterministic evidence supplied below — phoneme-level detections, timestamps, counts, and acoustic measurements already computed by this platform's audio pipeline. You do not have access to the audio itself.

This evidence is deterministic and exhaustive for this response: if something isn't in it, it didn't happen. Do not invent observations, words, timestamps, or patterns that are not present in the evidence. Every score, finding, and reasoning statement you produce must be traceable to a specific item in the evidence below."""

_TASK_STEPS = """Do the following, in order:

1. Analyze the evidence below.
2. Score each rubric this coach owns ({rubric_titles}) on a 1-5 scale (1 = poor, 2 = needs work, 3 = average, 4 = good, 5 = excellent), following the {coach_name} evaluation matrix — ground every score in specific evidence, never a holistic impression alone.
3. Using those scores and the evidence, generate the structured {coach_name} Coach JSON output below, following the schema exactly.
4. Produce a structured reasoning trace: for every score and every non-trivial claim in your coach output (each priority improvement, detailed finding, and recurring/behavioral pattern), cite the exact evidence (word, timestamp, count, or measurement) that justifies it."""

_OUTPUT_CONTRACT = """Return exactly one JSON object with these three top-level keys, in this order:

{{
  "scores": {{
{scores_skeleton}
  }},
  "coach_output": {schema},
  "reasoning_trace": [
    {{"conclusion": "<which score or coach_output finding this explains>",
      "evidence_cited": ["<specific evidence items -- words, timestamps, counts>"],
      "reasoning": "<why this evidence supports that conclusion>"}}
    // one entry per score and per non-trivial coach_output finding
  ]
}}

Generate the sections in the order shown — scores first, then coach_output, then reasoning_trace: score before you narrate, narrate before you explain why.

Some facts will legitimately repeat across coach_output's own sub-sections (e.g. the same mispronounced word may appear in both priority_improvements and detailed_findings) — that duplication is intentional per this platform's schema and should not be removed or consolidated.

Be concise but informative in every "reasoning" and "why_it_matters" field — one to two sentences, evidence-backed, no filler.

Return ONLY the JSON object. No prose, no markdown fences, no text before or after it."""

_EVIDENCE_BLOCK = """Evidence for session `{session_id}` ({coach_name} packet — Level 1 Timeline events and Level 2 Analytics already segregated for this coach):

{packet_json}"""


def _human_list(items: Sequence[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _scores_skeleton(rubrics: Sequence[str]) -> str:
    lines = [
        f'    "{rubric}": {{"score": <1-5 integer>, "reasoning": "<one sentence, cite specific evidence>"}}'
        for rubric in rubrics
    ]
    return ",\n".join(lines)


def _build_prompt(
    *,
    teacher_name: str,
    role_title: str,
    coach_name: str,
    rubrics: Sequence[str],
    rubric_titles: str,
    schema: str,
    packet: Dict[str, Any],
    session_id: str,
    extra_schema_notes: str = "",
) -> str:
    schema_with_notes = schema + (f"\n\n{extra_schema_notes}" if extra_schema_notes else "")
    sections = [
        _ROLE_PREAMBLE.format(teacher_name=teacher_name, role_title=role_title),
        _TASK_STEPS.format(rubric_titles=rubric_titles, coach_name=coach_name),
        _OUTPUT_CONTRACT.format(scores_skeleton=_scores_skeleton(rubrics), schema=schema_with_notes),
        _EVIDENCE_BLOCK.format(
            session_id=session_id,
            coach_name=coach_name,
            packet_json=json.dumps(packet, indent=2, ensure_ascii=False),
        ),
    ]
    return "\n\n".join(sections)


def build_articulation_prompt(packet: Dict[str, Any], session_id: str) -> str:
    """Build the complete prompt for the Articulation Coach (MiniMax M3)
    from an articulation coach packet (backend.feedback.coach_packets.
    build_articulation_packet()'s output -- Pronunciation + MTI evidence).

    Returns a ready-to-send prompt string; pass it directly to
    TeacherProvider.generate_articulation().
    """
    return _build_prompt(
        teacher_name=ARTICULATION_TEACHER,
        role_title="Articulation Coach",
        coach_name="articulation",
        rubrics=ARTICULATION_RUBRICS,
        rubric_titles=_human_list(["Pronunciation", "MTI"]),
        schema=ARTICULATION_SCHEMA,
        packet=packet,
        session_id=session_id,
    )


def build_delivery_prompt(packet: Dict[str, Any], session_id: str) -> str:
    """Build the complete prompt for the Delivery Coach (MiMo-v2.5) from a
    delivery coach packet (backend.feedback.coach_packets.
    build_delivery_packet()'s output -- Fluency + Intonation + Engagement,
    plus rhythm, evidence).

    Returns a ready-to-send prompt string; pass it directly to
    TeacherProvider.generate_delivery().
    """
    return _build_prompt(
        teacher_name=DELIVERY_TEACHER,
        role_title="Delivery Coach",
        coach_name="delivery",
        rubrics=DELIVERY_RUBRICS,
        rubric_titles=_human_list(["Fluency", "Intonation", "Engagement"]),
        schema=DELIVERY_SCHEMA,
        packet=packet,
        session_id=session_id,
        extra_schema_notes=DELIVERY_EVIDENCE_SHAPES,
    )
