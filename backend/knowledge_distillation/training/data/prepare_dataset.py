"""
Converts the knowledge-distillation dataset into HF-style conversational
JSONL for supervised fine-tuning (Part 8, stage 2).

NOT what training.trainer.dataset.py reads. That module reads
training.data.student_dataset's output directly (one already-split file
per coach, {"input": {"level1", "level2"}, "target": {...}}) -- this
module's combined-across-coaches conversations.jsonl/splits/ output is no
longer anything downstream consumes. It's kept because it's a valid,
independently useful representation (HF chat-format, all coaches in one
file) and deleting a working, tested module wasn't asked for -- but if
you're looking for what actually feeds a training run, that's
student_dataset.py, not this file.

Formatting itself is delegated entirely to training.prompts.build_conversation()
-- the coach-template-based formatter (Part 9) -- rather than duplicated
here. That used not to be true: this module briefly had its own parallel
inline formatter, which meant every new teacher_output field (most
recently evaluation_analysis and score_reasoning) had to be added in two
places or silently go stale in one of them. Given training.prompts already
existed as the more principled, template-file-based formatter, keeping a
second implementation here served no purpose but drift risk -- so this
module is now a thin loop: DistillationSample in, build_conversation() out,
written to JSONL. See training.prompts.prompt_builder's docstring for
exactly what a conversation contains.

Usage:
    python -m backend.knowledge_distillation.training.data.prepare_dataset
    python -m backend.knowledge_distillation.training.data.prepare_dataset --coach articulation
    python -m backend.knowledge_distillation.training.data.prepare_dataset --output path/to/out.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from . import COACHES, PREPARED_DATASET_PATH, load_coach_samples
from ..prompts import build_conversation

logger = logging.getLogger(__name__)

_LOG_EVERY = 500

# Re-exported for backward compatibility -- callers that imported
# to_conversation directly from this module (rather than from
# training.prompts, its new home) keep working unchanged.
to_conversation = build_conversation


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
