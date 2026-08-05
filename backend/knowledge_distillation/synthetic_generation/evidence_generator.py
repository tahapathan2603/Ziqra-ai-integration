"""
Orchestrator for ONE synthetic evidence packet.

    blueprint -> PromptBuilder -> Provider -> parse -> SyntheticEvidencePacket

One blueprint in, one packet out. Deliberately does not know about retries
(that's provider.py's job for transient failures), validation, or diversity
(pipeline.py's job) — it just requests generation and returns the parsed
result, or raises.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from ...llm.client import extract_json
from .config import SyntheticGenerationConfig
from .exceptions import EvidenceParsingError
from .prompt_builder import PromptBuilder
from .provider import Provider
from .schemas import GenerationMetadata, SpeakerBlueprint, SyntheticEvidencePacket

logger = logging.getLogger(__name__)

_REQUIRED_TOP_LEVEL_KEYS = ("level1", "level2")


def _generate_session_id() -> str:
    """Timestamp + short uuid4 hex, matching the real pipeline's format
    (see backend/distillation/dataset_builder.generate_session_id) without
    importing that module — this file has no other reason to depend on it."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"session_{timestamp}_{uuid.uuid4().hex[:8]}"


class EvidenceGenerator:
    """Turns one SpeakerBlueprint into one typed SyntheticEvidencePacket."""

    def __init__(
        self,
        provider: Provider,
        config: SyntheticGenerationConfig,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._prompt_builder = prompt_builder or PromptBuilder(config)

    def generate(self, blueprint: SpeakerBlueprint) -> SyntheticEvidencePacket:
        """Generate one evidence packet for `blueprint`.

        Raises:
            PromptBuildError: the prompt could not be built.
            NonRetryableProviderError / ProviderError: see provider.py.
            EvidenceParsingError: the response had no valid {level1, level2} JSON.
        """
        prompt = self._prompt_builder.build(blueprint)
        raw_response = self._provider.generate(prompt)

        parsed = extract_json(raw_response)
        if parsed is None or not isinstance(parsed, dict):
            raise EvidenceParsingError(
                f"No JSON object found in the provider's response for blueprint "
                f"'{blueprint.blueprint_id}'. Response: {raw_response!r}"
            )
        missing = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in parsed]
        if missing:
            raise EvidenceParsingError(
                f"Response for blueprint '{blueprint.blueprint_id}' is missing top-level "
                f"key(s) {missing}. Got keys: {list(parsed.keys())}"
            )

        return SyntheticEvidencePacket(
            session_id=_generate_session_id(),
            blueprint=blueprint,
            level1=parsed["level1"],
            level2=parsed["level2"],
            metadata=GenerationMetadata(
                model=self._config.model,
                prompt_version=self._config.prompt_version,
                schema_version=self._config.schema_version,
            ),
        )


def packet_from_authored(
    blueprint: SpeakerBlueprint,
    level1: dict,
    level2: dict,
    config: SyntheticGenerationConfig,
    author: str = "claude-code-session",
) -> SyntheticEvidencePacket:
    """Wrap directly-authored Level1/Level2 content into a typed packet,
    skipping the provider/prompt round-trip entirely.

    Exists for the no-API-key path: there is no Anthropic key configured in
    this project and no way for a script to call this Claude Code session
    as an HTTP-callable model, so packets are authored inline (by the agent,
    in-session) rather than fetched from a provider. Everything downstream
    of "has a SyntheticEvidencePacket" -- validator.py, diversity_filter.py,
    pipeline.py's output writing -- is unchanged and unaware of the
    difference.
    """
    return SyntheticEvidencePacket(
        session_id=_generate_session_id(),
        blueprint=blueprint,
        level1=level1,
        level2=level2,
        metadata=GenerationMetadata(
            model=author,
            prompt_version=config.prompt_version,
            schema_version=config.schema_version,
        ),
    )
