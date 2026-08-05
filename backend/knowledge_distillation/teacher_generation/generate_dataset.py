"""
Pipeline-orchestration script: runs the Teacher Runner across every session
in the accepted evidence dataset and writes each coach's raw teacher
response to disk.

This is deliberately NOT part of the layered packet_builder/prompt_builder/
provider/teacher_runner abstraction (Parts 1-4) -- it is the "loop over the
whole dataset and save results" step those parts explicitly left out of
scope. It performs no parsing and no validation: every output record's
"raw_response" is exactly what TeacherRunner returned, unmodified. That is
intentional -- validation is a separate, later stage.

Resumable: sessions already present in an output file are skipped on the
next run, so an interrupted run (rate limit, network blip, Ctrl-C) can be
restarted without re-paying for already-completed calls.

Aborts the whole run immediately on NonRetryableTeacherError (account-level
failure -- no credits, bad key): continuing would just burn through the
remaining sessions producing the same failure every time. A ordinary
TeacherProviderError (retries exhausted on one session) is logged and
skipped -- that session stays absent from the output file and is retried
automatically the next time this script runs.

The provider is injectable (defaults to a real TeacherProvider ->
MiniMax M3 / MiMo-v2.5) so the same script works whether the caller is
running against real teacher models or against ClaudeTeacherProvider
(claude_teacher.py) -- see that module's docstring for why it exists.
Each output record is tagged "generated_by" with the provider's `.name`,
so a later run against a different provider is distinguishable from
(and safe to intermix with) an earlier one.

Usage:
    venv/bin/python3 -m backend.knowledge_distillation.teacher_generation.generate_dataset [--limit N]
"""

import argparse
import json
import logging
import os
from typing import Any, Optional, Set

from .exceptions import NonRetryableTeacherError, TeacherProviderError
from .packet_builder import build_all
from .teacher_runner import TeacherRunner

logger = logging.getLogger(__name__)

DEFAULT_PACKETS_PATH = "backend/knowledge_distillation/synthetic_generation/datasets/packets/packets.jsonl"
DEFAULT_OUT_DIR = "backend/knowledge_distillation/teacher_generation/datasets/raw_responses"


def _already_done(path: str) -> Set[str]:
    """Session ids already present in an output file, so a rerun can skip
    them rather than re-calling a teacher model for work already done."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["session_id"])
    return done


def run(
    packets_path: str = DEFAULT_PACKETS_PATH,
    out_dir: str = DEFAULT_OUT_DIR,
    limit: Optional[int] = None,
    provider: Optional[Any] = None,
) -> None:
    """Run both teachers over every session in `packets_path` (or the first
    `limit` sessions), appending raw responses to
    `out_dir/articulation_raw.jsonl` and `out_dir/delivery_raw.jsonl`.

    Args:
        provider: Injected into TeacherRunner -- defaults to a real
            TeacherProvider (MiniMax M3 / MiMo-v2.5). Pass
            ClaudeTeacherProvider() (claude_teacher.py) to generate labels
            locally instead. Must expose a `.name` attribute, recorded on
            every output record as "generated_by".

    Raises:
        NonRetryableTeacherError: propagated immediately from whichever
            teacher hit an account-level failure -- the run stops there;
            already-written lines are untouched, rerun this function to
            resume once the underlying problem (credentials, credits) is
            fixed.
    """
    os.makedirs(out_dir, exist_ok=True)
    articulation_path = os.path.join(out_dir, "articulation_raw.jsonl")
    delivery_path = os.path.join(out_dir, "delivery_raw.jsonl")

    done_articulation = _already_done(articulation_path)
    done_delivery = _already_done(delivery_path)
    logger.info(
        "Resuming: %d articulation / %d delivery sessions already done.",
        len(done_articulation), len(done_delivery),
    )

    runner = TeacherRunner(provider=provider)
    generated_by = getattr(provider, "name", "minimax-m3+mimo-v2.5")
    processed = 0

    with open(articulation_path, "a", encoding="utf-8") as art_f, \
            open(delivery_path, "a", encoding="utf-8") as del_f:
        for pair in build_all(packets_path):
            if limit is not None and processed >= limit:
                break
            sid = pair.session_id

            if sid not in done_articulation:
                try:
                    raw = runner.run_articulation(pair.articulation, sid)
                except NonRetryableTeacherError:
                    logger.error("Non-retryable articulation failure on %s -- aborting run.", sid)
                    raise
                except TeacherProviderError as e:
                    logger.warning("Articulation failed for %s, will retry on next run: %s", sid, e)
                else:
                    art_f.write(json.dumps({"session_id": sid, "generated_by": generated_by, "raw_response": raw}, ensure_ascii=False) + "\n")
                    art_f.flush()

            if sid not in done_delivery:
                try:
                    raw = runner.run_delivery(pair.delivery, sid)
                except NonRetryableTeacherError:
                    logger.error("Non-retryable delivery failure on %s -- aborting run.", sid)
                    raise
                except TeacherProviderError as e:
                    logger.warning("Delivery failed for %s, will retry on next run: %s", sid, e)
                else:
                    del_f.write(json.dumps({"session_id": sid, "generated_by": generated_by, "raw_response": raw}, ensure_ascii=False) + "\n")
                    del_f.flush()

            processed += 1
            if processed % 25 == 0:
                logger.info("Processed %d sessions.", processed)

    logger.info("Done. Processed %d sessions this run.", processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sessions.")
    parser.add_argument("--packets-path", type=str, default=DEFAULT_PACKETS_PATH)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--provider", choices=["claude", "real"], default="claude",
        help="'claude' (default): local, no network, see claude_teacher.py. "
             "'real': live MiniMax M3 / MiMo-v2.5 calls via provider.py.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.provider == "claude":
        from .claude_teacher import ClaudeTeacherProvider
        selected_provider = ClaudeTeacherProvider()
    else:
        selected_provider = None  # TeacherRunner's own default: real TeacherProvider

    run(packets_path=args.packets_path, out_dir=args.out_dir, limit=args.limit, provider=selected_provider)
