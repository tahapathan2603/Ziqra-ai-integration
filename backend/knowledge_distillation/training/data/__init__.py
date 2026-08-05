"""
Data preparation for knowledge-distillation training (Part 8).

Note on naming: "the distillation dataset" doesn't exist yet as one merged
file on disk. Two artifacts from teacher_generation are joined on
session_id to produce it:

    teacher_generation/datasets/{coach}/{coach}.jsonl
        -- the coach packet: Level 1 Timeline + Level 2 Analytics, already
           segregated per coach by backend.feedback.coach_packets.
           build_coach_packets() (see teacher_generation/prompt_builder.py's
           docstring, which calls this exact thing "Level 1 Timeline events
           and Level 2 Analytics already segregated for this coach" -- this
           module reuses that same vocabulary for the same object).
            +
    teacher_generation/datasets/raw_responses/{coach}_raw.jsonl
        -- {session_id, generated_by, raw_response}, where raw_response is
           a JSON string of exactly {scores, coach_output, reasoning_trace}
           -- the teacher's structured output for that same session.
            =
    DistillationSample(session_id, coach, input, teacher_output)

This is the one thing every script in this package needs, so it lives here
rather than being duplicated three times: quality_check.py, prepare_dataset.py,
and split_dataset.py (indirectly, via prepare_dataset.py's output) all load
through `load_coach_samples`/`load_all_samples`.

    packet_path + raw_response_path
            |
            v
    load_coach_samples() / load_all_samples()   (this module)
            |
            v
    DistillationSample                    -- quality_check.py inspects
            |
            v
    prepare_dataset.py                    -- Sample -> HF chat-format record
            |
            v
    split_dataset.py                      -- train / validation / test JSONL

Adding a third coach later means one new COACHES entry -- nothing else in
this package hardcodes "articulation"/"delivery".
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ....feedback.coach_packets import build_coach_packets  # noqa: F401 -- see NOTE below
from ...teacher_generation.prompt_builder import ARTICULATION_RUBRICS, DELIVERY_RUBRICS

logger = logging.getLogger(__name__)

# NOTE on the build_coach_packets import above: this module does not call
# it -- coach packets are already built and on disk by the time this
# package runs. The import exists only so a schema drift in that function
# breaks an import here loudly, rather than this package silently reading
# a coach-packet shape that no longer matches production.

_DATA_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _DATA_DIR.parent
_KD_ROOT = _TRAINING_DIR.parent  # backend/knowledge_distillation/
_TEACHER_GEN_DATASETS = _KD_ROOT / "teacher_generation" / "datasets"

# Where this package's own outputs go by default.
PREPARED_DATASET_PATH = _DATA_DIR / "datasets" / "prepared" / "conversations.jsonl"
SPLITS_DIR = _DATA_DIR / "datasets" / "splits"


class DatasetLoadError(Exception):
    """The dataset on disk could not be loaded/joined as expected -- a
    missing file, unparseable JSON line, or a teacher response whose
    session_id has no matching coach packet."""


@dataclass(frozen=True)
class CoachDatasetPaths:
    """Where one coach's two source files live, and which rubrics its
    teacher_output.scores is expected to carry."""

    coach: str
    packet_path: Path
    raw_response_path: Path
    rubrics: Tuple[str, ...]


COACHES: Dict[str, CoachDatasetPaths] = {
    "articulation": CoachDatasetPaths(
        coach="articulation",
        packet_path=_TEACHER_GEN_DATASETS / "articulation" / "articulation.jsonl",
        raw_response_path=_TEACHER_GEN_DATASETS / "raw_responses" / "articulation_raw.jsonl",
        rubrics=ARTICULATION_RUBRICS,
    ),
    "delivery": CoachDatasetPaths(
        coach="delivery",
        packet_path=_TEACHER_GEN_DATASETS / "delivery" / "delivery.jsonl",
        raw_response_path=_TEACHER_GEN_DATASETS / "raw_responses" / "delivery_raw.jsonl",
        rubrics=DELIVERY_RUBRICS,
    ),
}


@dataclass(frozen=True)
class DistillationSample:
    """One training example: one session, scored by one coach.

    Attributes:
        session_id: The session this sample is for.
        coach: "articulation" | "delivery" (or any later COACHES key).
        input: The coach packet body (Level 1 Timeline + Level 2 Analytics,
            already segregated for this coach) -- everything the teacher
            model was shown, minus the redundant top-level session_id.
        teacher_output: {"scores", "coach_output", "reasoning_trace"},
            parsed from that session's raw_response.
        generated_by: Provenance tag from the raw response record (e.g.
            "claude"), carried through for traceability.
    """

    session_id: str
    coach: str
    input: Dict[str, Any]
    teacher_output: Dict[str, Any]
    generated_by: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read every line of `path` as a JSON object. Raises DatasetLoadError
    (never a bare exception) with the file and line number on the first
    malformed line -- this is a hard failure, not a data-quality nuance,
    since it means the file itself is broken."""
    if not path.exists():
        raise DatasetLoadError(f"Expected dataset file not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DatasetLoadError(f"{path}:{lineno}: invalid JSON ({e})") from e
    return records


def load_coach_samples(
    coach: str,
    coaches: Dict[str, CoachDatasetPaths] = COACHES,
    strict: bool = True,
    on_error: Optional[Callable[[str], None]] = None,
) -> Iterator[DistillationSample]:
    """Join one coach's packet file with its raw_response file on
    session_id, yielding a DistillationSample per successfully-joined pair.

    Every teacher response must have a matching coach packet and parseable
    JSON to become a sample -- a packet with NO response (e.g. a
    generate_dataset.py run that hasn't reached it yet) is simply not
    yielded, not an error, since "incomplete" is an expected transient
    dataset state.

    Args:
        coach: A key in `coaches` (e.g. "articulation").
        coaches: Coach -> paths/rubrics config; override in tests.
        strict: If True (default), any join problem raises
            DatasetLoadError immediately -- the right behavior for
            prepare_dataset.py, which cannot build a training example
            from a broken pair. If False, problems are reported via
            `on_error` (if given) and that record is skipped instead --
            the right behavior for quality_check.py, which must keep
            going to report every problem in one pass.
        on_error: Called with a human-readable message for each skipped
            problem when `strict=False`. Ignored when `strict=True`.

    Raises:
        DatasetLoadError: `coach` is unknown, a source file is missing or
            unparseable, or (`strict=True` only) a response's session_id
            has no matching packet / an unparseable raw_response.
    """
    if coach not in coaches:
        raise DatasetLoadError(f"Unknown coach '{coach}'. Known coaches: {sorted(coaches)}")
    cfg = coaches[coach]

    def fail(message: str) -> None:
        if strict:
            raise DatasetLoadError(message)
        if on_error is not None:
            on_error(message)

    packets_by_id = {
        r["session_id"]: r for r in _load_jsonl(cfg.packet_path) if "session_id" in r
    }
    response_records = _load_jsonl(cfg.raw_response_path)

    for resp in response_records:
        session_id = resp.get("session_id")
        if not session_id:
            fail(f"{cfg.raw_response_path}: record missing session_id: {resp!r}")
            continue

        packet = packets_by_id.get(session_id)
        if packet is None:
            fail(
                f"session '{session_id}' has a teacher response in {cfg.raw_response_path} "
                f"but no matching packet in {cfg.packet_path}"
            )
            continue

        try:
            teacher_output = json.loads(resp["raw_response"])
        except (TypeError, KeyError, json.JSONDecodeError) as e:
            fail(f"session '{session_id}': unparseable raw_response in {cfg.raw_response_path} ({e})")
            continue

        packet_body = {k: v for k, v in packet.items() if k != "session_id"}
        yield DistillationSample(
            session_id=session_id,
            coach=coach,
            input=packet_body,
            teacher_output=teacher_output,
            generated_by=resp.get("generated_by", "unknown"),
        )


def load_all_samples(
    coaches: Dict[str, CoachDatasetPaths] = COACHES,
    strict: bool = True,
    on_error: Optional[Callable[[str], None]] = None,
) -> Iterator[DistillationSample]:
    """`load_coach_samples` for every coach in `coaches`, in sorted-key order."""
    for coach in sorted(coaches):
        yield from load_coach_samples(coach, coaches, strict=strict, on_error=on_error)


__all__ = [
    "COACHES",
    "PREPARED_DATASET_PATH",
    "SPLITS_DIR",
    "CoachDatasetPaths",
    "DatasetLoadError",
    "DistillationSample",
    "load_all_samples",
    "load_coach_samples",
]
