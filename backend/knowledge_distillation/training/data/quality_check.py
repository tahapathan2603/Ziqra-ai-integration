"""
Quality check for the knowledge-distillation dataset (Part 8, inspection
only -- never modifies anything on disk).

For each coach, loads and joins its coach-packet + raw-response files (see
this package's __init__.py for exactly how), validates every sample's
schema, and prints a console report: sample counts, missing/invalid
records, duplicate session IDs, and every rubric's score distribution.
Optionally also writes the same report as JSON.

Usage:
    python -m backend.knowledge_distillation.training.data.quality_check
    python -m backend.knowledge_distillation.training.data.quality_check --coach articulation
    python -m backend.knowledge_distillation.training.data.quality_check --json-report report.json
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from . import COACHES, DatasetLoadError, DistillationSample, _load_jsonl, load_coach_samples

logger = logging.getLogger(__name__)

REQUIRED_TEACHER_OUTPUT_KEYS: tuple = ("scores", "coach_output", "reasoning_trace")
VALID_SCORE_VALUES = {1, 2, 3, 4, 5}
MAX_SCHEMA_EXAMPLES = 10  # cap how many bad-sample examples the report keeps in full


def _validate_sample(sample: DistillationSample, rubrics: tuple) -> List[str]:
    """Return every schema problem found in `sample` (empty list = valid).
    Pure inspection -- never raises, never mutates `sample`."""
    issues: List[str] = []

    if not sample.input:
        issues.append("input (coach packet) is empty")
    elif sample.input.get("coach") != sample.coach:
        issues.append(f"input.coach={sample.input.get('coach')!r} does not match coach {sample.coach!r}")

    missing_keys = [k for k in REQUIRED_TEACHER_OUTPUT_KEYS if k not in sample.teacher_output]
    if missing_keys:
        issues.append(f"teacher_output missing key(s): {missing_keys}")
        return issues  # nothing further can be checked without these

    scores = sample.teacher_output.get("scores") or {}
    for rubric in rubrics:
        entry = scores.get(rubric)
        if entry is None:
            issues.append(f"scores is missing rubric '{rubric}'")
            continue
        if entry.get("score") not in VALID_SCORE_VALUES:
            issues.append(f"scores.{rubric}.score={entry.get('score')!r} not in {sorted(VALID_SCORE_VALUES)}")
        if not entry.get("reasoning"):
            issues.append(f"scores.{rubric}.reasoning is empty")

    if not sample.teacher_output.get("coach_output"):
        issues.append("coach_output is missing/empty")
    if not sample.teacher_output.get("reasoning_trace"):
        issues.append("reasoning_trace is missing/empty")

    return issues


def _source_file_stats(path: Path) -> Dict[str, Any]:
    """Duplicate-session-id and record-count stats for one raw source
    file, independent of whether it successfully joins with its pair."""
    records = _load_jsonl(path)
    ids = [r.get("session_id") for r in records if r.get("session_id")]
    duplicates = sorted({sid for sid, count in Counter(ids).items() if count > 1})
    return {"total_records": len(records), "unique_session_ids": len(set(ids)), "duplicate_session_ids": duplicates}


def check_coach(coach: str) -> Dict[str, Any]:
    """Run every check for one coach and return a JSON-serializable report.

    Raises:
        DatasetLoadError: a source file is missing or contains a line that
            isn't valid JSON at all (a broken file, not a data-quality
            nuance -- see load_coach_samples's docstring for the
            strict/non-strict distinction this function relies on).
    """
    cfg = COACHES[coach]
    packet_stats = _source_file_stats(cfg.packet_path)
    response_stats = _source_file_stats(cfg.raw_response_path)

    join_errors: List[str] = []
    samples: List[DistillationSample] = list(
        load_coach_samples(coach, strict=False, on_error=join_errors.append)
    )

    score_distributions: Dict[str, Counter] = {r: Counter() for r in cfg.rubrics}
    generated_by = Counter()
    missing_coach_output = 0
    missing_reasoning_trace = 0
    schema_issues: Dict[str, List[str]] = {}
    reasoning_trace_lengths: List[int] = []

    for sample in samples:
        generated_by[sample.generated_by] += 1

        issues = _validate_sample(sample, cfg.rubrics)
        if issues:
            schema_issues[sample.session_id] = issues

        if not sample.teacher_output.get("coach_output"):
            missing_coach_output += 1
        trace = sample.teacher_output.get("reasoning_trace")
        if not trace:
            missing_reasoning_trace += 1
        else:
            reasoning_trace_lengths.append(len(trace))

        scores = sample.teacher_output.get("scores") or {}
        for rubric in cfg.rubrics:
            score = (scores.get(rubric) or {}).get("score")
            if score in VALID_SCORE_VALUES:
                score_distributions[rubric][score] += 1

    n = len(samples)
    return {
        "coach": coach,
        "total_samples": n,
        "packet_file": {"path": str(cfg.packet_path), **packet_stats},
        "raw_response_file": {"path": str(cfg.raw_response_path), **response_stats},
        "join_errors": join_errors,
        "invalid_schema_count": len(schema_issues),
        "invalid_schema_examples": dict(list(schema_issues.items())[:MAX_SCHEMA_EXAMPLES]),
        "missing_coach_output": missing_coach_output,
        "missing_reasoning_trace": missing_reasoning_trace,
        "generated_by": dict(generated_by),
        "avg_reasoning_trace_entries": (
            round(sum(reasoning_trace_lengths) / len(reasoning_trace_lengths), 2) if reasoning_trace_lengths else 0
        ),
        "score_distributions": {
            rubric: {
                "counts": dict(sorted(counter.items())),
                "min_band_share_pct": round(100 * min(counter.values()) / n, 1) if counter and n else 0.0,
            }
            for rubric, counter in score_distributions.items()
        },
    }


def _print_report(report: Dict[str, Any]) -> None:
    coach = report["coach"]
    print(f"\n{'=' * 60}")
    print(f"Coach: {coach}")
    print(f"{'=' * 60}")
    print(f"  Packet file:        {report['packet_file']['path']}")
    print(f"    records={report['packet_file']['total_records']}  "
          f"unique_session_ids={report['packet_file']['unique_session_ids']}  "
          f"duplicates={report['packet_file']['duplicate_session_ids'] or 'none'}")
    print(f"  Raw response file:  {report['raw_response_file']['path']}")
    print(f"    records={report['raw_response_file']['total_records']}  "
          f"unique_session_ids={report['raw_response_file']['unique_session_ids']}  "
          f"duplicates={report['raw_response_file']['duplicate_session_ids'] or 'none'}")
    print(f"  Joined samples:     {report['total_samples']}")
    print(f"  Join errors:        {len(report['join_errors'])}")
    for msg in report["join_errors"][:MAX_SCHEMA_EXAMPLES]:
        print(f"    - {msg}")

    print(f"  Invalid schema:     {report['invalid_schema_count']}")
    for session_id, issues in list(report["invalid_schema_examples"].items())[:5]:
        print(f"    - {session_id}: {issues}")

    print(f"  Missing coach_output:     {report['missing_coach_output']}")
    print(f"  Missing reasoning_trace:  {report['missing_reasoning_trace']}")
    print(f"  Avg reasoning_trace entries: {report['avg_reasoning_trace_entries']}")
    print(f"  generated_by:       {report['generated_by']}")

    print("  Score distributions:")
    for rubric, dist in report["score_distributions"].items():
        print(f"    {rubric:14s} {dist['counts']}  (min band = {dist['min_band_share_pct']}% of dataset)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coach", choices=sorted(COACHES) + ["all"], default="all", help="which coach to check")
    parser.add_argument("--json-report", type=Path, default=None, help="also write the full report as JSON to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    coaches = sorted(COACHES) if args.coach == "all" else [args.coach]
    reports = []
    for coach in coaches:
        try:
            report = check_coach(coach)
        except DatasetLoadError as e:
            print(f"FAILED to load dataset for coach '{coach}': {e}")
            raise SystemExit(1)
        reports.append(report)
        _print_report(report)

    print(f"\n{'=' * 60}")
    total = sum(r["total_samples"] for r in reports)
    total_bad = sum(r["invalid_schema_count"] for r in reports)
    print(f"TOTAL: {total} samples across {len(reports)} coach(es), {total_bad} with schema issues")

    if args.json_report:
        try:
            args.json_report.parent.mkdir(parents=True, exist_ok=True)
            args.json_report.write_text(json.dumps(reports, indent=2))
        except OSError as e:
            print(f"FAILED to write JSON report to {args.json_report}: {e}")
            raise SystemExit(1)
        print(f"JSON report written to {args.json_report}")


if __name__ == "__main__":
    main()
