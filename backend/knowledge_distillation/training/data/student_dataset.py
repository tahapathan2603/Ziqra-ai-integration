"""
Student training dataset preparation (Part 8, stage 4).

Converts the distillation dataset -- teacher_generation's coach packets
(teacher_generation/datasets/{coach}/) plus raw teacher responses
(teacher_generation/datasets/raw_responses/), already the project's single
source of truth for what a teacher said about a session, see this
package's __init__.py for exactly how they join -- into flat input/target
pairs, with no chat-template formatting applied:

    DistillationSample.input (coach packet)
            |
            v
    training.prompts.split_evidence()   -- Level 1 Timeline / Level 2
            |                               Analytics; the same split
            |                               training.prompts already uses
            v
    {"session_id", "coach",
     "input": {"level1": ..., "level2": ...},
     "target": {"evaluation_analysis": ..., "scores": ...,
                "score_reasoning": ..., "coach_output": ...,
                "reasoning_trace": ...}}
            |
            v
    datasets/student_dataset/{coach}/{train,validation,test}.jsonl

Reuses training.data's loader (the packet/raw_response join has exactly
one implementation) and training.prompts' evidence split (the Level 1/
Level 2 distinction has exactly one implementation) rather than
re-deriving either. `REQUIRED_TEACHER_OUTPUT_KEYS` -- this module's
definition of what "target" must contain -- is quality_check.py's
constant, not a second copy of the same five field names.

Note on "the distillation dataset must remain unchanged": this module
only reads it (via training.data's loader, strict mode -- a broken source
aborts the run rather than silently shipping a partial dataset) and never
writes to teacher_generation/datasets/ or its raw_responses/. It is not
physically duplicated into a datasets/distillation_dataset/ path -- see
this module's project-summary note for why.

Independent from training.trainer/: this produces a DIFFERENT
representation than training.data.prepare_dataset.py's conversations.jsonl
(HF chat-format {"messages": [...]}, which training.trainer.dataset.py
already consumes via a chat template). The two aren't reconciled by this
module -- which shape an eventual fine-tuning approach should read is a
decision for that stage.

Usage:
    python -m backend.knowledge_distillation.training.data.student_dataset
    python -m backend.knowledge_distillation.training.data.student_dataset --seed 7 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import COACHES, DatasetLoadError, DistillationSample, load_coach_samples
from .quality_check import REQUIRED_TEACHER_OUTPUT_KEYS
from .split_dataset import _split_group
from ..prompts import split_evidence

logger = logging.getLogger(__name__)

# "Configuration ... avoid hardcoding these values" -- same shape as
# split_dataset.py's DEFAULT_RATIOS/DEFAULT_SEED: named constants,
# overridable via CLI flags. Not a YAML file: this project's convention
# for a handful of scalar parameters is constants + argparse (see
# split_dataset.py, prepare_dataset.py), and a new config-file format for
# three ratios and a seed would be more machinery than the thing it
# configures.
DEFAULT_RATIOS: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train, validation, test
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "student_dataset"

_RATIO_SUM_TOLERANCE = 1e-6


class StudentDatasetError(Exception):
    """A ratio configuration is invalid, a sample is missing a required
    input/target field, or duplicate session_ids were found in one
    coach's source data."""


def _validate_record_or_raise(record: Dict[str, Any]) -> None:
    session_id = record["session_id"]
    if not record.get("input", {}).get("level1") and not record.get("input", {}).get("level2"):
        raise StudentDatasetError(f"session '{session_id}': input has neither level1 nor level2 evidence")
    target = record.get("target") or {}
    missing = [k for k in REQUIRED_TEACHER_OUTPUT_KEYS if not target.get(k)]
    if missing:
        raise StudentDatasetError(f"session '{session_id}': target missing/empty field(s): {missing}")


def build_student_record(sample: DistillationSample) -> Dict[str, Any]:
    """One DistillationSample -> one {"input": {level1, level2}, "target": {...}}
    pair. session_id/coach ride alongside for traceability, same as
    training.data.prepare_dataset's conversations -- a consumer that only
    wants "input"/"target" simply ignores them.

    Raises:
        StudentDatasetError: input or target is missing a required field
            -- see this module's docstring for what "required" means.
    """
    level1, level2 = split_evidence(sample.input)
    record = {
        "session_id": sample.session_id,
        "coach": sample.coach,
        "input": {"level1": level1, "level2": level2},
        "target": {key: sample.teacher_output.get(key) for key in REQUIRED_TEACHER_OUTPUT_KEYS},
    }
    _validate_record_or_raise(record)
    return record


def prepare_coach_records(coach: str) -> List[Dict[str, Any]]:
    """Every student record for one coach, each validated before being
    returned -- see build_student_record. Raises on the first invalid
    sample rather than silently dropping it: this is the final dataset a
    training run reads, not an inspection report."""
    return [build_student_record(sample) for sample in load_coach_samples(coach)]


def split_and_write(
    coach: str,
    records: List[Dict[str, Any]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, int]:
    """Split `records` into train/validation/test -- seeded and
    reproducible, no duplication across splits by construction (each
    record lands in exactly one partition; see split_dataset._split_group,
    reused here rather than reimplemented) -- and write each to
    {output_dir}/{coach}/{split}.jsonl.

    "Balanced distribution of speaking profiles across splits": the
    source blueprints (synthetic_generation.blueprint_generator) sample
    every speaking-profile dimension independently and uniformly, so a
    uniform random split already distributes them proportionally across
    train/validation/test without needing explicit stratification on top
    -- adding that would be tracking population balance the source
    generator already guarantees.
    """
    if abs(sum(ratios) - 1.0) > _RATIO_SUM_TOLERANCE:
        raise StudentDatasetError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    session_ids = [r["session_id"] for r in records]
    duplicates = sorted({sid for sid in session_ids if session_ids.count(sid) > 1})
    if duplicates:
        raise StudentDatasetError(f"duplicate session_id(s) in '{coach}' source data: {duplicates}")

    rng = random.Random(f"{seed}:{coach}")
    train, validation, test = _split_group(records, ratios, rng)

    coach_dir = output_dir / coach
    coach_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for name, group in (("train", train), ("validation", validation), ("test", test)):
        out_path = coach_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in group:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[name] = len(group)
    return counts


def prepare(
    coaches: Optional[List[str]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Dict[str, int]]:
    """Build and write the student dataset for every coach in `coaches`
    (default: all of COACHES). Returns {coach: {split: count}}.

    Raises:
        DatasetLoadError: propagated from load_coach_samples (strict mode
            -- a broken source file or an unresolvable packet/response
            pair aborts the whole run).
        StudentDatasetError: an invalid ratio, a sample missing a required
            field, or duplicate session_ids in one coach's source data.
    """
    coaches = coaches or sorted(COACHES)
    report: Dict[str, Dict[str, int]] = {}
    for coach in coaches:
        records = prepare_coach_records(coach)
        counts = split_and_write(coach, records, output_dir, ratios, seed)
        report[coach] = counts
        logger.info(
            "%s: %d total -> train=%d validation=%d test=%d",
            coach, len(records), counts["train"], counts["validation"], counts["test"],
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coach", choices=sorted(COACHES), action="append", dest="coaches", help="repeatable; defaults to every coach")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    try:
        report = prepare(coaches=args.coaches, output_dir=args.output_dir, ratios=ratios, seed=args.seed)
    except (StudentDatasetError, DatasetLoadError) as e:
        print(f"Student dataset preparation failed: {e}")
        raise SystemExit(1)

    print(f"\nStudent dataset written to {args.output_dir}")
    for coach, counts in report.items():
        total = sum(counts.values())
        print(f"  {coach}: {total} total -> " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
