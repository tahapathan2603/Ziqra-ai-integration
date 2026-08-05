"""
Pipeline orchestration only — no generation, validation, or diversity logic
of its own. Wires blueprint_generator -> evidence_generator -> validator ->
diversity_filter into the reject-and-repeat loop:

    Generate Blueprint -> Generate Evidence -> Validate -> Check Diversity
    -> Accept or Reject -> Repeat

Stops only once `config.target_dataset_size` packets have been ACCEPTED —
not generated. A packet that fails validation or diversity is discarded and
the loop simply draws a fresh blueprint; nothing is retried in place.

An account-level provider failure (NonRetryableProviderError) aborts the
whole run immediately rather than being treated as one more rejected
attempt — every subsequent call would hit the identical wall.

This module's only output is raw evidence (packets/packets.jsonl) — Level 1
+ Level 2, nothing derived from it. It does NOT call build_coach_packets()
and does NOT write articulation/delivery files; that used to happen here
and was removed because it duplicated
backend.knowledge_distillation.teacher_generation.packet_builder, which is
now the SOLE place that derives Articulation/Delivery coach packets from
accepted evidence. See that module's docstring for the full pipeline
staging.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .blueprint_generator import BlueprintGenerator
from .config import SyntheticGenerationConfig
from .diversity_filter import DiversityFilter
from .evidence_generator import EvidenceGenerator, packet_from_authored
from .exceptions import NonRetryableProviderError, SyntheticGenerationError
from .schemas import PipelineStats, SpeakerBlueprint, SyntheticEvidencePacket
from .validator import Validator

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs the generate/validate/diversity-filter loop to
    `config.target_dataset_size` accepted packets."""

    def __init__(
        self,
        blueprint_generator: BlueprintGenerator,
        evidence_generator: EvidenceGenerator,
        validator: Validator,
        diversity_filter: DiversityFilter,
        config: SyntheticGenerationConfig,
    ) -> None:
        self._blueprint_generator = blueprint_generator
        self._evidence_generator = evidence_generator
        self._validator = validator
        self._diversity_filter = diversity_filter
        self._config = config

        self.rejected_validation = 0
        self.rejected_diversity = 0
        self.provider_failures = 0
        self.total_attempts = 0
        self.aborted_reason = ""

    def run_iter(self) -> Iterator[SyntheticEvidencePacket]:
        """Yield accepted packets one at a time until
        `config.target_dataset_size` have been yielded, or the run is
        aborted (NonRetryableProviderError) or exhausts its attempt budget.

        Pure orchestration: no file I/O. `run()` wraps this with output
        writing for the common case.
        """
        accepted = 0
        max_attempts = self._config.target_dataset_size * self._config.max_attempt_factor

        while accepted < self._config.target_dataset_size:
            if self.total_attempts >= max_attempts:
                self.aborted_reason = (
                    f"max_attempt_factor exceeded ({self.total_attempts} attempts, "
                    f"{accepted}/{self._config.target_dataset_size} accepted)"
                )
                logger.error(self.aborted_reason)
                return

            self.total_attempts += 1
            blueprint = self._blueprint_generator.generate()

            try:
                packet = self._evidence_generator.generate(blueprint)
            except NonRetryableProviderError as e:
                self.aborted_reason = f"non-retryable provider failure: {e}"
                logger.error("Aborting run: %s", self.aborted_reason)
                return
            except SyntheticGenerationError as e:
                self.provider_failures += 1
                logger.warning("Blueprint %s failed generation: %s", blueprint.blueprint_id, e)
                continue

            validation_result = self._validator.validate(packet)
            if not validation_result.accepted:
                self.rejected_validation += 1
                logger.info(
                    "Blueprint %s rejected by validation (%d issue(s)): %s",
                    blueprint.blueprint_id, len(validation_result.issues),
                    "; ".join(f"{i.category.value}/{i.field}: {i.message}" for i in validation_result.issues[:3]),
                )
                continue

            diversity_result = self._diversity_filter.check(blueprint)
            if not diversity_result.accepted:
                self.rejected_diversity += 1
                logger.info("Blueprint %s rejected by diversity filter: %s", blueprint.blueprint_id, diversity_result.reason)
                continue

            self._diversity_filter.record(blueprint)
            accepted += 1
            if accepted % 50 == 0:
                logger.info("accepted %d/%d packets", accepted, self._config.target_dataset_size)
            yield packet

    def run(self, out_dir: Optional[str] = None) -> PipelineStats:
        """Run to completion, optionally streaming accepted packets to
        `out_dir/packets/packets.jsonl` (the raw evidence packet, Level 1 +
        Level 2 only). Deriving Articulation/Delivery coach packets from
        this file is teacher_generation.packet_builder's job, not this
        method's — see the module docstring.
        """
        start = time.time()
        accepted_count = 0

        if out_dir is None:
            for _ in self.run_iter():
                accepted_count += 1
        else:
            packets_dir = os.path.join(out_dir, "packets")
            os.makedirs(packets_dir, exist_ok=True)

            with open(os.path.join(packets_dir, "packets.jsonl"), "w", encoding="utf-8") as packets_f:
                for packet in self.run_iter():
                    packets_f.write(json.dumps(packet.to_dict(), ensure_ascii=False) + "\n")
                    accepted_count += 1

        return PipelineStats(
            accepted=accepted_count,
            rejected_validation=self.rejected_validation,
            rejected_diversity=self.rejected_diversity,
            provider_failures=self.provider_failures,
            total_attempts=self.total_attempts,
            elapsed_seconds=round(time.time() - start, 2),
            aborted_reason=self.aborted_reason,
        )


# No default_pipeline() here on purpose: there is no concrete Provider in
# this codebase to wire one to (see provider.py's module docstring) --
# building one now would mean silently defaulting to whatever ZIQRA_TEACHER_*
# points at (Qwen), which this phase must never call. ingest_authored() is
# the actual entry point for this phase. Pipeline.run() remains usable
# directly once a real Provider exists: Pipeline(blueprint_generator=...,
# evidence_generator=EvidenceGenerator(your_provider, config), ...).


def _warm_diversity_filter_from_existing(diversity_filter: DiversityFilter, packets_path: str) -> int:
    """Reconstruct diversity_filter's counts from an existing
    packets.jsonl, so a later ingest_authored() call knows what's already
    been accepted without needing separate persisted state -- the output
    file itself is the single source of truth. Returns how many existing
    accepted packets were folded in."""
    if not os.path.exists(packets_path):
        return 0
    count = 0
    with open(packets_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            bp = record["blueprint"]
            blueprint = SpeakerBlueprint(
                blueprint_id=bp["blueprint_id"], pronunciation=bp["pronunciation"],
                mti_severity=bp["mti_severity"], pace=bp["pace"], filler_frequency=bp["filler_frequency"],
                rhythm=bp["rhythm"], intonation=bp["intonation"], engagement=bp["engagement"],
                confidence=bp["confidence"],
            )
            diversity_filter.record(blueprint)
            count += 1
    return count


def ingest_authored(
    authored: List[Tuple[SpeakerBlueprint, Dict[str, Any], Dict[str, Any]]],
    config: SyntheticGenerationConfig,
    out_dir: str,
    author: str = "claude-code-session",
) -> Dict[str, Any]:
    """Validate, diversity-filter, and append a batch of directly-authored
    (blueprint, level1, level2) triples to `out_dir/packets/packets.jsonl`
    (raw evidence only -- see module docstring for why this no longer also
    derives/writes Articulation/Delivery packets).

    For the no-API-key path: content is authored inline (by the agent, in a
    Claude Code session) rather than fetched from a provider -- see
    evidence_generator.packet_from_authored's docstring. Safe to call
    repeatedly across many small batches/turns: diversity state is rebuilt
    from the existing packets.jsonl on every call (no separate state file
    to go stale or get out of sync), and packets are appended, not
    overwritten.

    Args:
        authored: (blueprint, level1, level2) triples for ONE batch.
        config: Resolved settings (thresholds, target size for diversity quotas).
        out_dir: Same directory Pipeline.run() would be given.
        author: Stamped onto GenerationMetadata.model in place of a provider name.

    Returns:
        {"accepted": int, "rejected": int, "total_accepted_so_far": int,
         "rejections": [{"blueprint_id": str, "category": str, "reason": str}, ...]}
    """
    validator = Validator(config)
    diversity_filter = DiversityFilter(config)

    packets_dir = os.path.join(out_dir, "packets")
    os.makedirs(packets_dir, exist_ok=True)
    packets_path = os.path.join(packets_dir, "packets.jsonl")

    total_accepted_so_far = _warm_diversity_filter_from_existing(diversity_filter, packets_path)

    accepted = 0
    rejections: List[Dict[str, str]] = []

    with open(packets_path, "a", encoding="utf-8") as packets_f:
        for blueprint, level1, level2 in authored:
            packet = packet_from_authored(blueprint, level1, level2, config, author=author)

            validation_result = validator.validate(packet)
            if not validation_result.accepted:
                for issue in validation_result.issues:
                    rejections.append({
                        "blueprint_id": blueprint.blueprint_id,
                        "category": issue.category.value,
                        "reason": f"{issue.field}: {issue.message}",
                    })
                continue

            diversity_result = diversity_filter.check(blueprint)
            if not diversity_result.accepted:
                rejections.append({
                    "blueprint_id": blueprint.blueprint_id,
                    "category": "diversity",
                    "reason": diversity_result.reason,
                })
                continue

            diversity_filter.record(blueprint)
            packets_f.write(json.dumps(packet.to_dict(), ensure_ascii=False) + "\n")
            accepted += 1
            total_accepted_so_far += 1

    return {
        "accepted": accepted,
        "rejected": len(authored) - accepted,
        "total_accepted_so_far": total_accepted_so_far,
        "rejections": rejections,
    }
