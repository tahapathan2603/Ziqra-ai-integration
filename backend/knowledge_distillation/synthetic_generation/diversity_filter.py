"""
Tracks the distribution of ACCEPTED speaker blueprints and rejects a
candidate that would overrepresent an already-common trait or an exact
repeated combination.

Not validation — a rejected packet here is correct, internally-consistent
evidence; it's just one the dataset doesn't need more of right now.
Deliberately simple: two Counters, no embeddings, no semantic similarity.
Sufficient at the current target size; if it ever needs to scale to a much
larger dataset, the natural next step is bucket counts persisted across
runs (see config.py's docstring on TARGET_DATASET_SIZE), not a rewrite of
this file.
"""

from collections import Counter
from typing import Dict

from .config import SyntheticGenerationConfig
from .schemas import DiversityResult, SpeakerBlueprint


class DiversityFilter:
    """Call `check()` before accepting a packet, `record()` only after it's
    actually accepted (validation passed too) — a packet that fails
    validation must never consume diversity headroom."""

    def __init__(self, config: SyntheticGenerationConfig) -> None:
        self._config = config
        self._dimension_counts: Dict[str, Counter] = {
            dimension: Counter() for dimension in config.blueprint_dimensions
        }
        self._fingerprint_counts: Counter = Counter()
        self._accepted_count = 0

    def check(self, blueprint: SpeakerBlueprint) -> DiversityResult:
        """Whether `blueprint` would keep the dataset within its diversity
        quotas if accepted. Does not mutate state -- call `record()` after
        the packet is fully accepted."""
        for dimension, level in blueprint.as_dict().items():
            levels = self._config.blueprint_dimensions[dimension]
            cap = self._cap_for_dimension(len(levels))
            count = self._dimension_counts[dimension][level]
            if count >= cap:
                return DiversityResult(
                    accepted=False,
                    reason=f"{dimension}={level} already at {count}/{cap} accepted packets",
                )

        fingerprint = blueprint.fingerprint()
        repeats = self._fingerprint_counts[fingerprint]
        if repeats >= self._config.max_exact_fingerprint_repeats:
            return DiversityResult(
                accepted=False,
                reason=f"exact blueprint combination repeated {repeats} times already",
            )

        return DiversityResult(accepted=True)

    def record(self, blueprint: SpeakerBlueprint) -> None:
        """Fold an accepted blueprint into the tracked distribution."""
        for dimension, level in blueprint.as_dict().items():
            self._dimension_counts[dimension][level] += 1
        self._fingerprint_counts[blueprint.fingerprint()] += 1
        self._accepted_count += 1

    def _cap_for_dimension(self, n_levels: int) -> int:
        """Max accepted count for one level of a dimension with `n_levels`
        levels: an even split of the target, with headroom so 8
        independently-tracked quotas don't cause pathological rejection
        rates late in a run."""
        fair_share = self._config.target_dataset_size / n_levels
        return max(1, round(fair_share * self._config.diversity_overrepresentation_factor))

    def distribution(self) -> Dict[str, Dict[str, int]]:
        """Current accepted-count distribution per dimension, for reporting."""
        return {dimension: dict(counter) for dimension, counter in self._dimension_counts.items()}
