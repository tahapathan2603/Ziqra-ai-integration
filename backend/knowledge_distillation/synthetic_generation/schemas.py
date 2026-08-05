"""
Typed models for synthetic evidence generation — everything entering or
leaving this module.

SpeakerBlueprint uses explicit named fields, not a generic dict bag, per the
brief's "avoid passing unstructured dictionaries throughout the pipeline."
Level1/Level2 stay plain dicts: their internal shape IS the externally-owned
production schema (audio_analyzer.py / dataset_builder.py), not this
module's internal plumbing, so typing them here would just duplicate a
contract this module doesn't own.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SpeakerBlueprint:
    """The intended speaking behaviour for one synthetic evidence packet.

    Describes ONLY speech characteristics the deterministic audio pipeline
    actually measures — no interview role, seniority, or question
    difficulty; those are context the audio pipeline never analyzes.

    Attributes:
        blueprint_id: Stable identifier for this blueprint within a run.
        pronunciation: weak | average | strong | excellent
        mti_severity: none | light | moderate | heavy
        pace: slow | moderate | fast
        filler_frequency: minimal | moderate | frequent
        rhythm: steady | uniform | erratic
        intonation: flat | moderate | expressive
        engagement: disengaged | neutral | engaging
        confidence: low | moderate | high
    """

    blueprint_id: str
    pronunciation: str
    mti_severity: str
    pace: str
    filler_frequency: str
    rhythm: str
    intonation: str
    engagement: str
    confidence: str

    def as_dict(self) -> Dict[str, str]:
        """The 8 speaking-behaviour dimensions as dimension -> level,
        excluding blueprint_id. Used by diversity_filter.py for tracking."""
        return {
            "pronunciation": self.pronunciation,
            "mti_severity": self.mti_severity,
            "pace": self.pace,
            "filler_frequency": self.filler_frequency,
            "rhythm": self.rhythm,
            "intonation": self.intonation,
            "engagement": self.engagement,
            "confidence": self.confidence,
        }

    def fingerprint(self) -> Tuple[str, ...]:
        """The exact 8-dimension combination, order-stable, for exact-repeat
        detection in diversity_filter.py."""
        d = self.as_dict()
        return tuple(d[k] for k in sorted(d))


@dataclass(frozen=True)
class GenerationMetadata:
    """Provenance for one generated evidence packet."""

    model: str
    prompt_version: str
    schema_version: str
    generated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticEvidencePacket:
    """One synthetic session's deterministic evidence — this module's only
    output.

    `level1`/`level2` match the real production shapes exactly (see
    backend/distillation/dataset_builder.py's build_timeline()/
    build_features()): level1 carries audio_metadata, speech_chunks,
    sentences, words, detected_phonemes, acoustic_contours; level2 carries
    the five analyzer blocks {fluency, pronunciation, mti, intonation,
    engagement} that backend.feedback.coach_packets.build_coach_packets()
    consumes verbatim.

    Attributes:
        session_id: Synthetic session identifier.
        blueprint: The speaking-behaviour blueprint this packet realizes.
        level1: The Level 1 timeline dict.
        level2: The Level 2 analysis dict.
        metadata: Provenance (model, versions, timestamp).
    """

    session_id: str
    blueprint: SpeakerBlueprint
    level1: Dict[str, Any]
    level2: Dict[str, Any]
    metadata: GenerationMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "blueprint": asdict(self.blueprint),
            "level1": self.level1,
            "level2": self.level2,
            "metadata": self.metadata.to_dict(),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ValidationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ValidationCategory(str, Enum):
    """Which of validator.py's five checks an issue came from."""

    SCHEMA = "schema"
    RANGE = "range"
    LOGICAL = "logical"
    TIMELINE_CONSISTENCY = "timeline_consistency"
    PRODUCTION_COMPATIBILITY = "production_compatibility"


@dataclass(frozen=True)
class ValidationIssue:
    category: ValidationCategory
    field: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"category": self.category.value, "field": self.field, "message": self.message}


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status is ValidationStatus.ACCEPTED

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "issues": [i.to_dict() for i in self.issues]}


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiversityResult:
    accepted: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineStats:
    """Run-level outcome, returned by pipeline.py."""

    accepted: int
    rejected_validation: int
    rejected_diversity: int
    provider_failures: int
    total_attempts: int
    elapsed_seconds: float
    aborted_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
