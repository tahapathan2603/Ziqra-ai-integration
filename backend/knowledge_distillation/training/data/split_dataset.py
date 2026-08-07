"""
Deterministic train / validation / test split of the prepared
conversational dataset (Part 8, stage 3).

NOT what training.trainer.dataset.py reads -- see prepare_dataset.py's
docstring (this module splits ITS output). training.data.student_dataset.py
does its own splitting directly against the distillation dataset, one file
per coach, and that's what actually feeds a training run.

Splits each coach's records independently by the same ratios, then
combines and shuffles the three splits -- this is what "preserves coach
balance": if the input is 2000 articulation + 2000 delivery records at an
80/10/10 split, train/validation/test each get the same 50/50 coach mix,
not an accidental skew from one coach happening to sort earlier.

Reproducible: every shuffle is seeded (`--seed`, default 42) from a
per-coach `random.Random`, and each coach's records are sorted by
session_id before shuffling -- so the split is a pure function of
(input file contents, ratios, seed), independent of the order records
happen to appear in the file. Every record lands in exactly one split
(no duplication): each coach group is partitioned into three disjoint
slices covering every record.

Usage:
    python -m backend.knowledge_distillation.training.data.split_dataset
    python -m backend.knowledge_distillation.training.data.split_dataset --seed 7
    python -m backend.knowledge_distillation.training.data.split_dataset --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import PREPARED_DATASET_PATH, SPLITS_DIR

logger = logging.getLogger(__name__)

DEFAULT_RATIOS: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train, validation, test
DEFAULT_SEED = 42
_RATIO_SUM_TOLERANCE = 1e-6


class SplitError(Exception):
    """The split configuration or input file is invalid -- ratios that
    don't sum to 1.0, a missing prepared dataset, or a malformed line."""


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SplitError(f"Prepared dataset not found: {path}. Run prepare_dataset.py first.")
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SplitError(f"{path}:{lineno}: invalid JSON ({e})") from e
    return records


def _split_group(
    records: List[Dict[str, Any]], ratios: Tuple[float, float, float], rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition one coach's records into (train, validation, test). Sorts
    by session_id first so the result doesn't depend on file order, then
    shuffles in place with `rng`. `test` gets everything not claimed by
    `train`/`validation`, so rounding can never drop or duplicate a
    record."""
    ordered = sorted(records, key=lambda r: r.get("session_id", ""))
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train = ordered[:n_train]
    validation = ordered[n_train:n_train + n_val]
    test = ordered[n_train + n_val:]
    return train, validation, test


def split(
    input_path: Path = PREPARED_DATASET_PATH,
    output_dir: Path = SPLITS_DIR,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, int]:
    """Split `input_path` into train/validation/test JSONL files under
    `output_dir`. Returns {"train": n, "validation": n, "test": n}.

    Raises:
        SplitError: `ratios` doesn't sum to 1.0, or `input_path` is
            missing/malformed.
    """
    if abs(sum(ratios) - 1.0) > _RATIO_SUM_TOLERANCE:
        raise SplitError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    records = _load_records(input_path)
    by_coach: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_coach.setdefault(record.get("coach", "unknown"), []).append(record)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for coach in sorted(by_coach):
        rng = random.Random(f"{seed}:{coach}")
        train, validation, test = _split_group(by_coach[coach], ratios, rng)
        splits["train"].extend(train)
        splits["validation"].extend(validation)
        splits["test"].extend(test)
        logger.info(
            "coach=%s total=%d -> train=%d validation=%d test=%d",
            coach, len(by_coach[coach]), len(train), len(validation), len(test),
        )

    # Interleave coaches within each split (still fully deterministic) so
    # a downstream reader that takes e.g. the first N lines doesn't get
    # one coach's records grouped together.
    combine_rng = random.Random(seed)
    for name in splits:
        combine_rng.shuffle(splits[name])

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for name, group in splits.items():
        out_path = output_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in group:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[name] = len(group)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=PREPARED_DATASET_PATH, help="prepared conversations.jsonl")
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR, help="where to write train/validation/test.jsonl")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    try:
        counts = split(args.input, args.output_dir, ratios, args.seed)
    except SplitError as e:
        print(f"Split failed: {e}")
        raise SystemExit(1)

    print(f"Wrote splits to {args.output_dir}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
