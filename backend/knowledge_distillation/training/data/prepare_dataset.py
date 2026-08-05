"""
Converts the knowledge-distillation dataset into HF-style conversational
JSONL for supervised fine-tuning (Part 8, stage 2).

Each DistillationSample (see this package's __init__.py) becomes one
two-turn conversation:

    user:      which coach, plus the full coach packet (Level 1 Timeline +
               Level 2 Analytics) as verbatim JSON -- exactly the evidence
               that coach's teacher model was shown.
    assistant: the teacher's Scores, Coach Output, and Reasoning Trace, each
               as verbatim JSON.

Every field from `input` and `teacher_output` is dumped through
json.dumps(..., indent=2) -- nothing is summarized, reformatted, or
dropped, so no information is lost converting to this format.

Usage:
    python -m backend.knowledge_distillation.training.data.prepare_dataset
    python -m backend.knowledge_distillation.training.data.prepare_dataset --coach articulation
    python -m backend.knowledge_distillation.training.data.prepare_dataset --output path/to/out.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import COACHES, PREPARED_DATASET_PATH, DistillationSample, load_coach_samples

logger = logging.getLogger(__name__)

_LOG_EVERY = 500


def _format_user_turn(sample: DistillationSample) -> str:
    evidence = json.dumps(sample.input, indent=2, ensure_ascii=False)
    return f"Coach: {sample.coach.title()}\n\nEvidence (Level 1 Timeline + Level 2 Analytics):\n{evidence}"


def _format_assistant_turn(sample: DistillationSample) -> str:
    scores = json.dumps(sample.teacher_output.get("scores"), indent=2, ensure_ascii=False)
    coach_output = json.dumps(sample.teacher_output.get("coach_output"), indent=2, ensure_ascii=False)
    reasoning_trace = json.dumps(sample.teacher_output.get("reasoning_trace"), indent=2, ensure_ascii=False)
    return f"Scores:\n{scores}\n\nCoach Output:\n{coach_output}\n\nReasoning Trace:\n{reasoning_trace}"


def to_conversation(sample: DistillationSample) -> Dict[str, Any]:
    """One DistillationSample -> one Hugging Face chat-format training
    record: {"messages": [{"role": ..., "content": ...}, ...]}, the
    standard shape for `tokenizer.apply_chat_template()` / TRL's
    SFTTrainer. `session_id`/`coach` ride alongside for traceability and
    for split_dataset.py's coach-stratified splitting; a trainer that only
    wants "messages" simply ignores the rest."""
    return {
        "session_id": sample.session_id,
        "coach": sample.coach,
        "messages": [
            {"role": "user", "content": _format_user_turn(sample)},
            {"role": "assistant", "content": _format_assistant_turn(sample)},
        ],
    }


def prepare(coaches: Optional[List[str]] = None, output_path: Path = PREPARED_DATASET_PATH) -> int:
    """Write one conversational-format JSONL line per sample, for every
    coach in `coaches` (default: all of COACHES). Returns the number of
    samples written.

    Raises:
        DatasetLoadError: propagated from load_coach_samples (strict mode
            -- a broken source file or an unresolvable packet/response
            pair aborts the whole run rather than silently shipping a
            partial training set).
    """
    coaches = coaches or sorted(COACHES)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for coach in coaches:
            for sample in load_coach_samples(coach):
                f.write(json.dumps(to_conversation(sample), ensure_ascii=False) + "\n")
                count += 1
                if count % _LOG_EVERY == 0:
                    logger.info("Prepared %d samples", count)

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--coach", choices=sorted(COACHES), action="append", dest="coaches",
        help="repeatable; defaults to every coach in COACHES",
    )
    parser.add_argument("--output", type=Path, default=PREPARED_DATASET_PATH, help="output JSONL path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    count = prepare(coaches=args.coaches, output_path=args.output)
    print(f"Prepared {count} training samples -> {args.output}")


if __name__ == "__main__":
    main()
