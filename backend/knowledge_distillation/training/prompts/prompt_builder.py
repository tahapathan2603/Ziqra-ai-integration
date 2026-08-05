"""
Prompt Builder for student training (Part 9).

Turns one DistillationSample (backend.knowledge_distillation.training.data)
into one complete Hugging Face chat-format training conversation:

    DistillationSample.input (coach packet)
            |
            v
    split_evidence()               -- Level 1 Timeline (timestamped events)
                                       + Level 2 Analytics (aggregate
                                       measurements); see its docstring for
                                       exactly how that split is derived
            |
            v
    {coach}_template.txt            -- filled in as the user turn
            |
    DistillationSample.teacher_output
            |
            v
    build_assistant_turn()          -- Scores + Coach Output + Reasoning
                                        Trace, verbatim, as the assistant turn
            |
            v
    {"session_id", "coach",
     "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Independent of any trainer: this module only produces plain `messages`
dicts. It imports nothing from torch, transformers, or peft, and nothing
here loads a model or runs training -- the (not-yet-implemented) trainer
module will call `build_conversation()` per sample the same way it will
eventually call a tokenizer, but that wiring doesn't exist yet.

Integrates with training.data (Part 8) by consuming its DistillationSample
type and its `load_coach_samples`/`load_all_samples` loaders directly,
rather than re-deriving the packet/raw_response join here -- that join has
exactly one implementation, in training.data.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from ..data import COACHES, DistillationSample, load_all_samples, load_coach_samples

_TEMPLATES_DIR = Path(__file__).resolve().parent
_TEMPLATE_FILES: Dict[str, Path] = {
    "articulation": _TEMPLATES_DIR / "articulation_template.txt",
    "delivery": _TEMPLATES_DIR / "delivery_template.txt",
}

_missing_templates = set(COACHES) - set(_TEMPLATE_FILES)
if _missing_templates:
    # Fail at import time, not at first use -- a coach added to
    # training.data.COACHES without a matching template here is a setup
    # bug, not a per-sample data problem.
    raise PromptBuildError(f"No template file mapping for coach(es): {sorted(_missing_templates)}")

_LEVEL1_PLACEHOLDER = "{LEVEL1_TIMELINE}"
_LEVEL2_PLACEHOLDER = "{LEVEL2_ANALYTICS}"

_TIMESTAMP_KEYS = {"start", "end", "timestamp"}


class PromptBuildError(Exception):
    """A conversation could not be built -- an unknown coach, or a missing
    template file."""


def _is_timeline_event(value: Any) -> bool:
    """True if `value` is a non-empty list containing at least one dict
    with a timestamp-like key ("start"/"end"/"timestamp") -- i.e. a list
    of WHEN-something-happened events, as opposed to a scalar measurement
    or classification."""
    return isinstance(value, list) and any(
        isinstance(item, dict) and _TIMESTAMP_KEYS & item.keys() for item in value
    )


def split_evidence(packet: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split one coach packet into (level1_timeline, level2_analytics).

    The coach packet (backend.feedback.coach_packets.build_coach_packets()'s
    output) does not carry a literal top-level level1/level2 split -- that
    distinction collapsed upstream: packet_builder.py passes only level2
    into build_coach_packets(), so no raw Level 1 timeline survives as a
    separate object. What DOES survive, inside each rubric, is a mix of
    timestamped events (mispronounced_words, MTI patterns, flat_sections --
    all "when did this happen") and aggregate measurements alongside them
    (accuracy percentages, rubric scores, counts, wpm -- all "what did we
    measure overall"). This function recovers that same distinction by
    walking each rubric's fields and bucketing them with
    `_is_timeline_event`, generically -- no rubric or coach name is
    hardcoded, so a new coach's packet splits the same way with no changes
    here.

    Every field ends up in exactly one of the two returned dicts (nothing
    duplicated, nothing dropped).
    """
    timeline: Dict[str, Any] = {}
    analytics: Dict[str, Any] = {}

    for key, value in packet.items():
        if not isinstance(value, dict):
            # "coach" and any other top-level non-rubric scalar -- not an
            # event list, so it's analytics by definition.
            analytics[key] = value
            continue

        timeline_fields = {field: v for field, v in value.items() if _is_timeline_event(v)}
        analytics_fields = {field: v for field, v in value.items() if field not in timeline_fields}

        if timeline_fields:
            timeline[key] = timeline_fields
        if analytics_fields:
            analytics[key] = analytics_fields

    return timeline, analytics


def load_template(coach: str) -> str:
    """Read `{coach}_template.txt` from this package.

    Raises:
        PromptBuildError: `coach` isn't a known template, or its file is
            missing from disk.
    """
    if coach not in _TEMPLATE_FILES:
        raise PromptBuildError(f"No prompt template for coach '{coach}'. Known coaches: {sorted(_TEMPLATE_FILES)}")
    path = _TEMPLATE_FILES[coach]
    if not path.exists():
        raise PromptBuildError(f"Template file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_user_turn(sample: DistillationSample) -> str:
    """Fill `sample`'s coach template with its Level 1 Timeline / Level 2
    Analytics evidence, both as verbatim indented JSON."""
    template = load_template(sample.coach)
    timeline, analytics = split_evidence(sample.input)
    return template.replace(
        _LEVEL1_PLACEHOLDER, json.dumps(timeline, indent=2, ensure_ascii=False)
    ).replace(
        _LEVEL2_PLACEHOLDER, json.dumps(analytics, indent=2, ensure_ascii=False)
    )


def build_assistant_turn(sample: DistillationSample) -> str:
    """The teacher's Scores, Coach Output, and Reasoning Trace, each as
    verbatim JSON -- nothing summarized, reformatted, or dropped."""
    scores = json.dumps(sample.teacher_output.get("scores"), indent=2, ensure_ascii=False)
    coach_output = json.dumps(sample.teacher_output.get("coach_output"), indent=2, ensure_ascii=False)
    reasoning_trace = json.dumps(sample.teacher_output.get("reasoning_trace"), indent=2, ensure_ascii=False)
    return f"Scores:\n{scores}\n\nCoach Output:\n{coach_output}\n\nReasoning Trace:\n{reasoning_trace}"


def build_conversation(sample: DistillationSample) -> Dict[str, Any]:
    """One DistillationSample -> one HF chat-format training record."""
    return {
        "session_id": sample.session_id,
        "coach": sample.coach,
        "messages": [
            {"role": "user", "content": build_user_turn(sample)},
            {"role": "assistant", "content": build_assistant_turn(sample)},
        ],
    }


def build_conversations(coach: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield `build_conversation()` for every sample of `coach` (or every
    coach in COACHES if `coach` is None), streaming straight from
    training.data's loader.

    Raises:
        DatasetLoadError: propagated from training.data's loader -- a
            missing source file or an unresolvable packet/response pair.
        PromptBuildError: propagated from build_conversation -- an unknown
            coach or missing template.
    """
    samples = load_coach_samples(coach) if coach is not None else load_all_samples()
    for sample in samples:
        yield build_conversation(sample)


__all__ = [
    "PromptBuildError",
    "build_assistant_turn",
    "build_conversation",
    "build_conversations",
    "build_user_turn",
    "load_template",
    "split_evidence",
]
