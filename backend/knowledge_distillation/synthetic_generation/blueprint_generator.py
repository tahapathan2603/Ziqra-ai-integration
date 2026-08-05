"""
Generates a SpeakerBlueprint: the intended speaking behaviour for one
synthetic evidence packet.

Deliberately dumb and cheap — no timestamps, no metrics, no scores, no
Level 1/2 values, and no call to a provider. Each of the 8 dimensions
(config.BLUEPRINT_DIMENSIONS) is drawn independently and uniformly at
random, so combinations vary freely (a blueprint CAN pair "strong
pronunciation" with "disengaged" — that's a real, valid speaking profile,
not a contradiction). Balance across the dataset is diversity_filter.py's
job, not this one's: rejecting an overrepresented draw and asking for
another is cheaper and simpler than this generator trying to plan ahead.
"""

import random
from typing import Optional

from .config import SyntheticGenerationConfig
from .schemas import SpeakerBlueprint


class BlueprintGenerator:
    """Draws random SpeakerBlueprints from config.BLUEPRINT_DIMENSIONS."""

    def __init__(self, config: SyntheticGenerationConfig, rng: Optional[random.Random] = None) -> None:
        self._config = config
        self._rng = rng or random.Random()
        self._next_id = 0

    def generate(self) -> SpeakerBlueprint:
        """Draw one blueprint, uniformly at random per dimension."""
        dims = self._config.blueprint_dimensions
        values = {name: self._rng.choice(levels) for name, levels in dims.items()}
        self._next_id += 1
        return SpeakerBlueprint(
            blueprint_id=f"bp_{self._next_id:06d}",
            pronunciation=values["pronunciation"],
            mti_severity=values["mti_severity"],
            pace=values["pace"],
            filler_frequency=values["filler_frequency"],
            rhythm=values["rhythm"],
            intonation=values["intonation"],
            engagement=values["engagement"],
            confidence=values["confidence"],
        )
