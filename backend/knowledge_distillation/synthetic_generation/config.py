"""
Single source of configuration for synthetic evidence generation. Nothing
in this module hardcodes a model name, threshold, range bound, or
dimension level anywhere else — they all resolve here.

RAW EVIDENCE ONLY: the generated Level 1 / Level 2 dataset carries no
scores, ratings, classifications, severity labels, or natural-language
observations of any kind — only raw measurements, counts, timestamps, and
event data, exactly what a deterministic analyzer would extract before any
interpretation. All scoring/interpretation/coaching happens later, in the
teacher-generation stage, over the coach packets this evidence eventually
produces via build_coach_packets(). This shapes every threshold below: there
is no SCORE_RANGE, and every "logical consistency" bound compares raw
numbers to each other, never a label to a metric.

No credentials, no endpoint, no LLMConfig: this phase has no concrete
Provider (see provider.py's docstring) and must not resolve or reference
any external model's configuration, including the project's own
ZIQRA_TEACHER_* (Qwen). `model`/`temperature`/etc. below are inert tuning
knobs stamped onto GenerationMetadata for whenever a real Provider exists —
they never drive a live call today.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

GENERATION_ENV_PREFIX = "ZIQRA_SYNTHGEN_"

# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------
DEFAULT_TEMPERATURE = 0.8  # varied, realistic evidence across ~2000 calls
DEFAULT_MAX_TOKENS = 4096  # one full Level1+Level2 JSON at reduced density
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 90.0

PROMPT_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Dataset size
# --------------------------------------------------------------------------
TARGET_DATASET_SIZE = 2000

# A blueprint that fails validation/diversity is discarded, not retried —
# the pipeline just draws a fresh blueprint (see pipeline.py). This caps
# total attempts so a systemically failing setup (bad prompt, exhausted
# credits that somehow isn't classified non-retryable) can't loop forever.
MAX_ATTEMPT_FACTOR = 8

# --------------------------------------------------------------------------
# Blueprint dimensions — the ONLY things a blueprint describes. Each is a
# plain qualitative band; concrete numbers are Claude's job when authoring
# evidence, not the blueprint's. Locked with the user: 8 dimensions, no
# role/experience/difficulty, no separate energy/clarity (energy lives
# inside intonation_expressiveness, clarity inside pronunciation/MTI --
# matching production's own analyzer categories 1:1).
# --------------------------------------------------------------------------
BLUEPRINT_DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "pronunciation": ("weak", "average", "strong", "excellent"),
    "mti_severity": ("none", "light", "moderate", "heavy"),
    "pace": ("slow", "moderate", "fast"),
    "filler_frequency": ("minimal", "moderate", "frequent"),
    "rhythm": ("steady", "uniform", "erratic"),
    "intonation": ("flat", "moderate", "expressive"),
    "engagement": ("disengaged", "neutral", "engaging"),
    "confidence": ("low", "moderate", "high"),
}

# --------------------------------------------------------------------------
# Level 1 density — acoustic_contours / detected_phonemes are dense in real
# capture (~0.03s hop, one phoneme entry per ~15 letters) but are NEVER read
# by build_coach_packets() (verified: it only reads Level 2). Asked at
# reduced density, the same tradeoff synthetic_data_generator.py already
# made deliberately ("no score/packet consumes them at full density") —
# keeps per-call tokens/latency predictable across a 2000-call run.
# --------------------------------------------------------------------------
CONTOUR_HOP_SECONDS = 0.25
MAX_DETECTED_PHONEMES_PER_WORD = 2

# --------------------------------------------------------------------------
# Range validation bounds. No SCORE_RANGE: the dataset carries no *_score
# fields at all (see module docstring) -- every remaining bound is on a raw
# measurement, never an evaluative number.
# --------------------------------------------------------------------------
FRACTION_RANGE: Tuple[float, float] = (0.0, 1.0)  # phoneme_accuracy, rhythm_score, stress_accuracy, word confidence
WPM_RANGE: Tuple[float, float] = (40.0, 260.0)
FILLERS_PER_MINUTE_RANGE: Tuple[float, float] = (0.0, 25.0)
DURATION_RANGE_SECONDS: Tuple[float, float] = (3.0, 90.0)

# --------------------------------------------------------------------------
# Logical-consistency thresholds. The dataset carries NO scores, ratings,
# classifications, or severity labels (see module docstring) -- every check
# here compares raw numbers/counts/timestamps against each other directly,
# never a label against a metric. (An earlier version of this file checked
# e.g. "classification == 'fast' vs wpm" and "delivery_label contains 'flat'
# vs pitch_range" -- both moot now that those label fields don't exist.)
# --------------------------------------------------------------------------
HIGH_ACCURACY_THRESHOLD = 0.90              # phoneme_accuracy considered "high"
MAX_MISPRONOUNCED_WORDS_AT_HIGH_ACCURACY = 2  # raw count allowed alongside high accuracy (no severity to weigh)

# NOTE on direction: in production's real (now-removed) monotonicity_score,
# LOW score meant MORE monotone/flat. Without that score, "how monotone" is
# read directly off monotone_sections' coverage of the recording instead.
MONOTONE_COVERAGE_RATIO_FOR_FLAT_CHECK = 0.5  # monotone_sections covering >= this fraction of duration counts as "genuinely flat"
FLAT_MAX_PITCH_RANGE_HZ = 120.0               # ...and genuinely-flat speech must then have a capped pitch_range

FILLER_RATE_CROSS_CHECK_TOLERANCE = 0.25  # fillers_per_minute vs filler_count/duration, fractional tolerance
FILLER_LIST_LENGTH_TOLERANCE = 1          # len(fillers[]) vs filler_count, absolute allowed difference

# --------------------------------------------------------------------------
# Timeline consistency (Level 1 vs Level 2) tolerances
# --------------------------------------------------------------------------
WPM_CROSS_CHECK_TOLERANCE = 0.30  # computed from level1.words vs level2 wpm, fractional tolerance
MIN_WORD_MATCH_RATIO = 0.8        # fraction of Level2-cited words that must actually appear in Level1.words

# --------------------------------------------------------------------------
# Diversity
# --------------------------------------------------------------------------
# A single dimension level is rejected once its share of ACCEPTED packets
# would exceed (1 / n_levels_for_that_dimension) * this factor. >1.0 gives
# headroom so 8 independently-tracked quotas don't cause pathological
# rejection rates late in a run.
DIVERSITY_OVERREPRESENTATION_FACTOR = 1.3

# The exact same 8-dimension combination is rejected after repeating this
# many times, regardless of individual dimension quotas.
MAX_EXACT_FINGERPRINT_REPEATS = 3


def _optional_float(name: str, default: float) -> float:
    value = os.environ.get(GENERATION_ENV_PREFIX + name)
    return float(value) if value else default


def _optional_int(name: str, default: int) -> int:
    value = os.environ.get(GENERATION_ENV_PREFIX + name)
    return int(value) if value else default


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    """Fully-resolved settings for one generation run."""

    model: str
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION

    target_dataset_size: int = TARGET_DATASET_SIZE
    max_attempt_factor: int = MAX_ATTEMPT_FACTOR

    blueprint_dimensions: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(BLUEPRINT_DIMENSIONS)
    )

    contour_hop_seconds: float = CONTOUR_HOP_SECONDS
    max_detected_phonemes_per_word: int = MAX_DETECTED_PHONEMES_PER_WORD

    fraction_range: Tuple[float, float] = FRACTION_RANGE
    wpm_range: Tuple[float, float] = WPM_RANGE
    fillers_per_minute_range: Tuple[float, float] = FILLERS_PER_MINUTE_RANGE
    duration_range_seconds: Tuple[float, float] = DURATION_RANGE_SECONDS

    high_accuracy_threshold: float = HIGH_ACCURACY_THRESHOLD
    max_mispronounced_words_at_high_accuracy: int = MAX_MISPRONOUNCED_WORDS_AT_HIGH_ACCURACY
    monotone_coverage_ratio_for_flat_check: float = MONOTONE_COVERAGE_RATIO_FOR_FLAT_CHECK
    flat_max_pitch_range_hz: float = FLAT_MAX_PITCH_RANGE_HZ
    filler_rate_cross_check_tolerance: float = FILLER_RATE_CROSS_CHECK_TOLERANCE
    filler_list_length_tolerance: int = FILLER_LIST_LENGTH_TOLERANCE

    wpm_cross_check_tolerance: float = WPM_CROSS_CHECK_TOLERANCE
    min_word_match_ratio: float = MIN_WORD_MATCH_RATIO

    diversity_overrepresentation_factor: float = DIVERSITY_OVERREPRESENTATION_FACTOR
    max_exact_fingerprint_repeats: int = MAX_EXACT_FINGERPRINT_REPEATS

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> "SyntheticGenerationConfig":
        """Build config from the environment.

        `model` defaults to ZIQRA_SYNTHGEN_MODEL if set, else
        "claude-code-session" (this phase's actual generator -- see the
        package docstring). Deliberately never falls back to
        ZIQRA_TEACHER_MODEL: that's Qwen's configuration, and this phase
        must not reference it, live call or not. Generation parameters read
        ZIQRA_SYNTHGEN_TEMPERATURE / _MAX_TOKENS / _MAX_RETRIES /
        _TIMEOUT_SECONDS / _TARGET_DATASET_SIZE when set.
        """
        resolved_model = model or os.environ.get(GENERATION_ENV_PREFIX + "MODEL") or "claude-code-session"
        return cls(
            model=resolved_model,
            temperature=_optional_float("TEMPERATURE", DEFAULT_TEMPERATURE),
            max_tokens=_optional_int("MAX_TOKENS", DEFAULT_MAX_TOKENS),
            max_retries=_optional_int("MAX_RETRIES", DEFAULT_MAX_RETRIES),
            timeout_seconds=_optional_float("TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            target_dataset_size=_optional_int("TARGET_DATASET_SIZE", TARGET_DATASET_SIZE),
        )
