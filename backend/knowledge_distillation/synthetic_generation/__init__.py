"""
Synthetic evidence generation: produces Level 1 + Level 2 deterministic
evidence packets that resemble the production audio pipeline's output,
for later knowledge-distillation stages to consume.

This package does NOT generate coaching outputs, teacher labels, or any
knowledge-distillation artifact — its only output is raw evidence packets
shaped exactly like production, so build_coach_packets() runs on them
completely unmodified.

Model responsibilities for this phase (do not blur these):
    Claude (this session)  -> generates Level 1 + Level 2 evidence (here)
    MiniMax M3              -> Articulation teacher outputs, later stage
    MiMo-v2.5                -> Delivery teacher outputs, later stage
    Gemma / Qwen              -> student models, fine-tuned later, on the
                                 finished dataset -- never used to generate it

    SpeakerBlueprint            (speaking behaviour only -- no role/
      (blueprint_generator)      seniority/difficulty; no metrics/scores)
            |
            v
    SyntheticEvidencePacket      (Level 1 timeline + Level 2 analysis,
      (evidence_generator /       raw measurements/events only -- see
       packet_from_authored)      config.py's docstring)
            |
            v
    ValidationResult             (schema / range / logical / timeline
      (validator)                 consistency / production compatibility)
            |
            v
    DiversityResult               (reject if overrepresented; lightweight
      (diversity_filter)           counter-based tracking)
            |
            v
    Accepted dataset              (pipeline.py's reject-and-repeat loop,
                                    stops at config.target_dataset_size
                                    ACCEPTED packets)

Usage — content authored in-session (the current path; see provider.py's
docstring for why there is no default LLM-backed path):
    from backend.knowledge_distillation.synthetic_generation import (
        SpeakerBlueprint, SyntheticGenerationConfig, ingest_authored,
    )

    config = SyntheticGenerationConfig.from_env()
    batch = [(blueprint, level1_dict, level2_dict), ...]
    result = ingest_authored(batch, config, out_dir="path/to/datasets")

Usage — once a real Provider exists (e.g. a genuine Claude API client):
    from backend.knowledge_distillation.synthetic_generation import (
        BlueprintGenerator, DiversityFilter, EvidenceGenerator, Pipeline,
        SyntheticGenerationConfig, Validator,
    )

    config = SyntheticGenerationConfig.from_env()
    pipeline = Pipeline(
        blueprint_generator=BlueprintGenerator(config),
        evidence_generator=EvidenceGenerator(your_provider, config),
        validator=Validator(config),
        diversity_filter=DiversityFilter(config),
        config=config,
    )
    stats = pipeline.run(out_dir="path/to/datasets")
"""

from .blueprint_generator import BlueprintGenerator
from .config import SyntheticGenerationConfig
from .diversity_filter import DiversityFilter
from .evidence_generator import EvidenceGenerator, packet_from_authored
from .exceptions import (
    EvidenceParsingError,
    NonRetryableProviderError,
    ProviderError,
    SyntheticGenerationError,
)
from .pipeline import Pipeline, ingest_authored
from .prompt_builder import PromptBuildError, PromptBuilder
from .provider import Provider
from .schemas import (
    DiversityResult,
    GenerationMetadata,
    PipelineStats,
    SpeakerBlueprint,
    SyntheticEvidencePacket,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationStatus,
)
from .validator import Validator

__all__ = [
    "BlueprintGenerator",
    "DiversityFilter",
    "DiversityResult",
    "EvidenceGenerator",
    "EvidenceParsingError",
    "GenerationMetadata",
    "NonRetryableProviderError",
    "Pipeline",
    "PipelineStats",
    "PromptBuildError",
    "PromptBuilder",
    "Provider",
    "ProviderError",
    "SpeakerBlueprint",
    "SyntheticEvidencePacket",
    "SyntheticGenerationConfig",
    "SyntheticGenerationError",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationResult",
    "ValidationStatus",
    "Validator",
    "ingest_authored",
    "packet_from_authored",
]
