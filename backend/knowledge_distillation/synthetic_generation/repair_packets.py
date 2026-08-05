"""
One-shot deterministic repair pass over the synthetic evidence dataset
(datasets/packets/packets.jsonl), fixing two evidence defects that collapse
downstream teacher score distributions for engagement and intonation:

  1. pitch_range lives in three DISJOINT per-blueprint clusters (flat
     20-50, moderate 80-140, expressive 160-260, with dead zones at 51-79
     and 141-159 Hz) and its top cluster is out-of-distribution vs
     production (real sessions: 27.8-128.8 Hz, median 70.6 --
     dataset/features/*.json; backend/tests/test_audio_pipeline_regressions.py:369
     asserts 40-150 Hz). Rescaled per-blueprint into one continuous,
     overlapping, production-realistic 28-130 Hz band.

  2. low_energy_sections -- the one piece of evidence that actually
     separates "engaging" from "disengaged" blueprints -- was placed
     independently of the engagement blueprint dimension (statistically
     inert: disengaged/neutral/engaging sessions were indistinguishable).
     Regenerated as a duration-relative COVERAGE FRACTION per blueprint
     level, mirroring how production's engagement_analyzer applies its
     COVERAGE_PENALTY_MAX over merged problem-section coverage
     (feature_extractors/audio/engagement/engagement_analyzer.py) rather
     than an absolute-seconds budget, which real session durations
     (4.8-22s) couldn't accommodate.

  3. severity ("high"/"medium") is backfilled onto both monotone_sections
     and low_energy_sections -- production always populates this
     (feature_extractors/audio/intonation/monotonicity.py:104,
     energy_variation.py:128) but the synthetic schema never did, leaving
     coach_packets._top_by_severity's truncation order arbitrary.

Touches ONLY level2.intonation. level1, and every other level2 block
(pronunciation, mti, fluency), are byte-identical before and after -- so
articulation and fluency teacher scores cannot shift; only the two
rubrics this repair targets can move. Verified by test_repair_packets.py.

Deterministic: every random draw is seeded from
sha256(f"{session_id}:{salt}"), so repairing the same PRISTINE input twice
produces byte-identical output. Rescaling is not its own inverse, though
(it maps one per-blueprint source band to a different target band), so
naively re-running against already-repaired output would rescale a second
time and corrupt the data. To make reruns safe regardless, this script
always repairs FROM the pristine `.bak` when one exists rather than from
whatever currently sits at the output path -- so "run it again" is always
idempotent, never cumulative. Only with `--no-backup` (no `.bak` ever
written) does a rerun repair its own prior output.

Usage:
    venv/bin/python3 -m backend.knowledge_distillation.synthetic_generation.repair_packets
    venv/bin/python3 -m backend.knowledge_distillation.synthetic_generation.repair_packets --input <path> --no-backup
"""

import argparse
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PACKETS_PATH = Path(__file__).parent / "datasets" / "packets" / "packets.jsonl"

# ---------------------------------------------------------------------------
# 1. Pitch range rescale -- per-blueprint linear remap, source cluster ->
#    overlapping production-realistic target band. Preserves within-band
#    rank (a session at the top of "flat"'s source range stays near the top
#    of "flat"'s target range).
# ---------------------------------------------------------------------------
PITCH_SOURCE_BANDS: Dict[str, Tuple[float, float]] = {
    "flat": (20.0, 50.0),
    "moderate": (80.0, 140.0),
    "expressive": (160.0, 260.0),
}
PITCH_TARGET_BANDS: Dict[str, Tuple[float, float]] = {
    "flat": (28.0, 58.0),
    "moderate": (52.0, 92.0),
    "expressive": (86.0, 130.0),
}

# ---------------------------------------------------------------------------
# 2. Low-energy coverage -- (n_sections range, coverage-fraction-of-duration
#    range) per engagement blueprint level. Overlapping ranges (not
#    disjoint) so band edges aren't another hard cliff.
# ---------------------------------------------------------------------------
LOW_ENERGY_BANDS: Dict[str, Tuple[Tuple[int, int], Tuple[float, float]]] = {
    "engaging": ((0, 1), (0.0, 0.12)),
    "neutral": ((1, 2), (0.10, 0.35)),
    "disengaged": ((2, 4), (0.30, 0.70)),
}
# A single low-energy/monotone stretch may never claim more than this
# fraction of the recording -- keeps placement physically plausible
# regardless of how short the session is.
MAX_COVERAGE_FRACTION = 0.9

# Section-level severity thresholds (mirrors the two-tier "high"/"medium"
# vocabulary production emits, without reproducing its acoustic thresholds
# -- there is no acoustic signal left in this synthetic schema to threshold
# against, so severity is read off each section's share of the recording
# instead). Both are fractions of `duration`, not absolute seconds: the
# delivery coach packet (backend/feedback/coach_packets.py) never exposes
# duration downstream of this repair, so a scorer reading flat_sections
# can only see counts and severities -- coverage-relative-to-duration
# must therefore be baked into severity HERE, at repair time, while
# duration is still available.
MONOTONE_HIGH_COVERAGE_RATIO = 0.6      # section covering >= this fraction of duration -> "high"
LOW_ENERGY_HIGH_COVERAGE_RATIO = 0.15   # a single low-energy section covering >= this fraction -> "high"


def _rng(session_id: str, salt: str) -> random.Random:
    """Deterministic per-(session, purpose) RNG -- same input always
    produces the same draws, so this script is idempotent."""
    digest = hashlib.sha256(f"{session_id}:{salt}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def rescale_pitch_range(raw_range: float, blueprint_level: str) -> int:
    """Linear remap of one pitch_range value from its source cluster into
    the corresponding overlapping target band. Rounds to int (the
    synthetic schema stores pitch fields as ints)."""
    src_lo, src_hi = PITCH_SOURCE_BANDS[blueprint_level]
    dst_lo, dst_hi = PITCH_TARGET_BANDS[blueprint_level]
    # Clamp defensively in case upstream data ever drifts outside its
    # documented source band.
    clamped = max(src_lo, min(src_hi, raw_range))
    fraction = (clamped - src_lo) / (src_hi - src_lo) if src_hi > src_lo else 0.5
    return round(dst_lo + fraction * (dst_hi - dst_lo))


def _place_sections(
    rng: random.Random, n: int, total_seconds: float, duration: float
) -> List[Dict[str, float]]:
    """Place `n` non-overlapping sections inside [0, duration] totalling
    ~total_seconds. Sections live in disjoint equal-width slots, so
    non-overlap is structural, not checked after the fact."""
    if n <= 0 or total_seconds <= 0 or duration <= 0:
        return []
    total_seconds = min(total_seconds, MAX_COVERAGE_FRACTION * duration)
    slot_width = duration / n
    weights = [rng.uniform(0.5, 1.5) for _ in range(n)]
    weight_sum = sum(weights)
    sections = []
    for i, weight in enumerate(weights):
        length = max(0.3, total_seconds * weight / weight_sum)
        length = min(length, slot_width * 0.9)
        slot_start = i * slot_width
        max_offset = max(0.0, slot_width - length)
        start = slot_start + rng.uniform(0, max_offset)
        end = start + length
        sections.append({"start": round(start, 2), "end": round(end, 2)})
    return sections


def _severity_for_low_energy(section: Dict[str, float], duration: float) -> str:
    if duration <= 0:
        return "medium"
    coverage_ratio = (section["end"] - section["start"]) / duration
    return "high" if coverage_ratio >= LOW_ENERGY_HIGH_COVERAGE_RATIO else "medium"


def _severity_for_monotone(section: Dict[str, float], duration: float) -> str:
    if duration <= 0:
        return "medium"
    coverage_ratio = (section.get("end", 0) - section.get("start", 0)) / duration
    return "high" if coverage_ratio >= MONOTONE_HIGH_COVERAGE_RATIO else "medium"


def repair_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new packet dict with level2.intonation repaired. `level1`
    and every other level2 block are the identical objects from `row`
    (not copies) -- repair_row never mutates them, so callers can rely on
    reference equality as a cheap "untouched" check."""
    session_id = row["session_id"]
    blueprint = row["blueprint"]
    level2 = row["level2"]
    intonation = level2["intonation"]
    duration = row["level1"]["audio_metadata"]["duration"]

    pitch_variation = intonation["pitch_variation"]
    average_pitch = pitch_variation["average_pitch"]
    new_range = rescale_pitch_range(pitch_variation["pitch_range"], blueprint["intonation"])
    new_min = round(average_pitch - new_range / 2)
    new_max = new_min + new_range  # exact: preserves max - min == pitch_range

    new_pitch_variation = {
        **pitch_variation,
        "min_pitch": new_min,
        "max_pitch": new_max,
        "pitch_range": new_range,
    }

    monotone_sections = [
        {**s, "severity": _severity_for_monotone(s, duration)}
        for s in intonation["monotonicity"]["monotone_sections"]
    ]

    (n_lo, n_hi), (frac_lo, frac_hi) = LOW_ENERGY_BANDS[blueprint["engagement"]]
    rng = _rng(session_id, "low_energy_sections")
    n_sections = rng.randint(n_lo, n_hi)
    coverage_fraction = 0.0 if n_sections == 0 else rng.uniform(frac_lo, frac_hi)
    placed = _place_sections(rng, n_sections, coverage_fraction * duration, duration)
    low_energy_sections = [{**s, "severity": _severity_for_low_energy(s, duration)} for s in placed]

    new_intonation = {
        **intonation,
        "pitch_variation": new_pitch_variation,
        "monotonicity": {**intonation["monotonicity"], "monotone_sections": monotone_sections},
        "energy_variation": {**intonation["energy_variation"], "low_energy_sections": low_energy_sections},
    }

    return {
        **row,
        "level2": {**level2, "intonation": new_intonation},
    }


def repair_file(input_path: Path, output_path: Path = None, backup: bool = True) -> int:
    """Repair every row in `input_path`, writing the result to
    `output_path` (defaults to `input_path`, i.e. in place). Returns the
    number of rows repaired.

    Writes via a temp file + atomic rename so a crash mid-run can't leave
    a truncated dataset. If `backup` and `output_path == input_path`, the
    pre-repair file is preserved once at `<input_path>.bak` -- a second
    run does not clobber an existing backup, so `.bak` always reflects
    the ORIGINAL unrepaired dataset.

    Source-of-truth resolution: if `<input_path>.bak` already exists, it
    -- not `input_path` -- is read as the source. This is what makes
    reruns idempotent instead of cumulative: `input_path` may already be
    repaired output from a previous run, but `.bak` is always the
    pristine original.
    """
    output_path = output_path or input_path
    backup_path = input_path.with_suffix(input_path.suffix + ".bak")

    if backup_path.exists():
        source_path = backup_path
        logger.info("Found existing %s -- repairing from the pristine original, not %s", backup_path, input_path)
    else:
        source_path = input_path
        if backup and output_path == input_path:
            backup_path.write_text(input_path.read_text())
            logger.info("Backed up pre-repair dataset to %s", backup_path)

    rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    repaired = [repair_row(row) for row in rows]

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in repaired:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, output_path)

    logger.info("Repaired %d packets -> %s", len(repaired), output_path)
    return len(repaired)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_PACKETS_PATH, help="packets.jsonl to repair")
    parser.add_argument("--output", type=Path, default=None, help="defaults to --input (in place)")
    parser.add_argument("--no-backup", action="store_true", help="skip writing a .bak of the pre-repair file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = repair_file(args.input, args.output, backup=not args.no_backup)
    print(f"Repaired {count} packets in {args.output or args.input}")


if __name__ == "__main__":
    main()
