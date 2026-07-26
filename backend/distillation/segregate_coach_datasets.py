"""
Split the combined synthetic audio dataset into the two per-coach training files.

Reads dataset/synthetic_audio_dataset.jsonl (one combined row per session:
{session_id, timeline, fluency, pronunciation, mti, intonation, engagement}) and
runs each row's five analysis blocks through the SAME packet builders the live
feedback path will use — backend/feedback/coach_packets.py — so the segregation,
cleaning, and compression rules (segmental vs. suprasegmental split, stress
quarantined, engagement reduced to score+level, emphasis reliability-gated,
evidence capped by severity) are applied identically here and at inference.

Writes two JSONL files, one packet per line, each carrying `session_id` so they
link back to the combined dataset and to each other:
    dataset/articulation_coach_dataset.jsonl  -> {session_id, coach, pronunciation, mti}
    dataset/delivery_coach_dataset.jsonl      -> {session_id, coach, fluency, intonation, rhythm, engagement}

This does NOT regenerate the dataset — it is a pure, re-runnable projection of the
existing combined file, so it can be re-run whenever coach_packets.py changes.
"""

import json
import logging
import os
from typing import Dict, Tuple

from backend.feedback.coach_packets import build_coach_packets

logger = logging.getLogger(__name__)

_ANALYSIS_KEYS = ("fluency", "pronunciation", "mti", "intonation", "engagement")


def segregate(combined_path: str, articulation_path: str, delivery_path: str) -> Tuple[int, Dict]:
    """Project the combined dataset into the two per-coach files.

    Returns (row_count, summary) where summary carries light sanity stats used by
    the CLI to report and self-check the output.
    """
    os.makedirs(os.path.dirname(articulation_path), exist_ok=True)
    n = 0
    art_field_bytes = 0
    dlv_field_bytes = 0
    stress_leaks = 0
    engagement_extra_keys = 0

    with open(combined_path, "r", encoding="utf-8") as fin, \
            open(articulation_path, "w", encoding="utf-8") as fa, \
            open(delivery_path, "w", encoding="utf-8") as fd:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            analysis = {k: row[k] for k in _ANALYSIS_KEYS}
            packets = build_coach_packets(analysis, session_id=row["session_id"])

            articulation = {"session_id": packets["session_id"], **packets["articulation"]}
            delivery = {"session_id": packets["session_id"], **packets["delivery"]}

            art_line = json.dumps(articulation, ensure_ascii=False)
            dlv_line = json.dumps(delivery, ensure_ascii=False)
            fa.write(art_line + "\n")
            fd.write(dlv_line + "\n")

            # Light self-checks (the coach_packets rules should already guarantee these).
            if "stress" in art_line.lower() or "stress" in dlv_line.lower():
                stress_leaks += 1
            if set(delivery["engagement"].keys()) != {"score", "level"}:
                engagement_extra_keys += 1

            art_field_bytes += len(art_line)
            dlv_field_bytes += len(dlv_line)
            n += 1

    summary = {
        "rows": n,
        "articulation_avg_bytes": round(art_field_bytes / n) if n else 0,
        "delivery_avg_bytes": round(dlv_field_bytes / n) if n else 0,
        "stress_leaks": stress_leaks,
        "engagement_blocks_with_extra_keys": engagement_extra_keys,
    }
    logger.info("Segregated %d rows -> %s, %s", n, articulation_path, delivery_path)
    return n, summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(project_root, "dataset")
    combined = os.path.join(dataset_dir, "synthetic_audio_dataset.jsonl")
    articulation = os.path.join(dataset_dir, "articulation_coach_dataset.jsonl")
    delivery = os.path.join(dataset_dir, "delivery_coach_dataset.jsonl")

    if not os.path.exists(combined):
        raise FileNotFoundError(f"Combined dataset not found: {combined}")

    n, summary = segregate(combined, articulation, delivery)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
