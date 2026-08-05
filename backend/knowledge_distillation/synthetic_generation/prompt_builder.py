"""
SpeakerBlueprint -> prompt string. Nothing else.

This file never calls a provider and never parses a response — it only
renders a blueprint's speaking-behaviour traits into the fixed instruction
template below, which specifies the exact Level 1 / Level 2 schema to fill
in.

RAW EVIDENCE ONLY (see config.py's module docstring for the full rationale):
no scores, no ratings, no classification labels, no free-text observations
of any kind. Every field below is a raw measurement, count, timestamp, or
event -- exactly what a deterministic analyzer extracts before any
interpretation. Interpretation, reasoning, and coaching are the
teacher-generation stage's job, over the coach packets this evidence
eventually produces via build_coach_packets() -- never this stage's.

ONE deliberate exception: mispronounced_words[].severity and MTI
vowel_patterns[]/consonant_patterns[].severity are kept, even though they
read like ratings. build_coach_packets() (backend/feedback/coach_packets.py)
reads them via HARD bracket indexing (`w["severity"]`, `p["severity"]`), not
`.get()` -- every other removed field degrades to None gracefully, but
these two crash the function outright if absent. Since "flows through
build_coach_packets() without modification" is non-negotiable and almost
every real packet has at least one mispronunciation or MTI pattern, keeping
these two is the only way both requirements can hold at once. Confirmed by
actually calling the function, not assumed.
"""

import logging

from .config import SyntheticGenerationConfig
from .exceptions import SyntheticGenerationError
from .schemas import SpeakerBlueprint

logger = logging.getLogger(__name__)


class PromptBuildError(SyntheticGenerationError):
    """A prompt could not be built for the given blueprint/config."""


_TRAIT_DESCRIPTIONS = {
    "pronunciation": {
        "weak": "Frequent, clear mispronunciations; phoneme accuracy on the low end.",
        "average": "Mostly clear with a handful of ordinary slips.",
        "strong": "Clear articulation with only rare, minor slips.",
        "excellent": "Crisp, accurate articulation throughout, essentially no errors.",
    },
    "mti_severity": {
        "none": "No detectable native-language interference patterns.",
        "light": "A few mild native-language-influenced substitutions.",
        "moderate": "A noticeable, recurring native-language accent pattern.",
        "heavy": "Strong, pervasive native-language interference affecting many words.",
    },
    "pace": {
        "slow": "Speaks deliberately; words per minute on the low end.",
        "moderate": "Comfortable, natural speaking speed.",
        "fast": "Speaks rapidly; words per minute on the high end.",
    },
    "filler_frequency": {
        "minimal": "Almost no filler words.",
        "moderate": "An occasional filler word.",
        "frequent": "Frequent filler words throughout.",
    },
    "rhythm": {
        "steady": "Natural, evenly-varied timing between words.",
        "uniform": "Oddly flat, metronomic timing.",
        "erratic": "Choppy, unpredictable timing with irregular gaps.",
    },
    "intonation": {
        "flat": "Very little pitch movement; low pitch range; delivery reads as monotone.",
        "moderate": "A normal, natural amount of pitch variation.",
        "expressive": "Lively, wide pitch variation.",
    },
    "engagement": {
        "disengaged": "Reads as low-energy and detached.",
        "neutral": "Reads as matter-of-fact, neither engaging nor flat.",
        "engaging": "Reads as energetic and engaging.",
    },
    "confidence": {
        "low": "Hesitant, more pauses and disfluencies than usual.",
        "moderate": "Reasonably steady delivery.",
        "high": "Assured, fluent delivery with few hesitations.",
    },
}

_LEVEL1_SCHEMA = """{
  "audio_metadata": {"duration": <float seconds>, "sample_rate": 16000, "detected_language": "en", "language_probability": <float 0-1>},
  "speech_chunks": [{"chunk_id": <int, starts at 1>, "start": <float>, "end": <float>}],
  "sentences": [{"text": <string>, "start": <float>, "end": <float>}],
  "words": [{"word": <string>, "start": <float>, "end": <float>, "confidence": <float 0-1>}],
  "detected_phonemes": [{"phoneme": "/x/", "start": <float>, "end": <float>}],
  "acoustic_contours": {
    "pitch_hz": [{"time": <float>, "hz": <float>, "voiced": <bool>}],
    "energy_rms": [{"time": <float>, "rms": <float>}]
  }
}"""

_LEVEL2_SCHEMA = """{
  "fluency": {
    "fillers": {"filler_count": <int>, "fillers_per_minute": <float>, "fillers": [{"word": <string>, "start": <float>, "end": <float>}]},
    "pauses": {"total_pauses": <int>, "pauses": [{"start": <float>, "end": <float>, "duration": <float>, "type": "natural"|"hesitation"|"long_pause"|"dead_air"|"filled_pause"}]},
    "speaking_speed": {"words_per_minute": <int>, "sentences_per_minute": <int>}
  },
  "pronunciation": {
    "phoneme_accuracy": <float 0-1>,
    "stress_accuracy": 1.0,
    "rhythm_score": <float 0-1>,
    "phoneme_errors": [{"word": <string>, "expected": "/x/"|null, "detected": "/y/"|null}],
    "detected_phonemes": [],
    "mispronounced_words": [{"word": <string>, "severity": "low"|"medium"|"high", "start": <float>, "end": <float>}],
    "stress_errors": [],
    "rhythm_issues": []
  },
  "mti": {
    "summary": {"vowel_pattern_issues": <int>, "consonant_pattern_issues": <int>, "stress_transfer_issues": 0},
    "vowel_patterns": [{"word": <string>, "timestamp": {"start": <float>, "end": <float>}, "issue": "vowel_substitution", "expected": "/x/", "detected": "/y/", "severity": "low"|"medium"|"high"}],
    "consonant_patterns": [{"word": <string>, "timestamp": {"start": <float>, "end": <float>}, "issue": "missing_aspiration"|"consonant_substitution", "expected": "/x/", "detected": "/y/", "severity": "low"|"medium"|"high"}],
    "stress_transfer": [],
    "speech_statistics": {"total_words_analyzed": <int>, "affected_words": <int>, "affected_percentage": <float>}
  },
  "intonation": {
    "pitch_variation": {"average_pitch": <float>, "min_pitch": <float>, "max_pitch": <float>, "pitch_range": <float>},
    "energy_variation": {"low_energy_sections": [{"start": <float>, "end": <float>}]},
    "monotonicity": {"monotone_sections": [{"start": <float>, "end": <float>}]},
    "emphasis": {"under_emphasized_words": [{"word": <string>, "timestamp": {"start": <float>, "end": <float>}}]}
  },
  "engagement": {}
}"""

_PROMPT_TEMPLATE = """You are generating ONE synthetic training example for an audio-analysis pipeline. You are inventing RAW DETERMINISTIC MEASUREMENTS ONLY -- no scores, no ratings, no classification labels, no natural-language observations, no coaching, no interpretation of any kind. Every field you produce must be a number, a timestamp, a count, or a bare factual event -- never a judgment about how good/bad/fast/flat something is. Interpretation happens in a later, separate stage; this stage never does it. The ONE exception is "severity" ("low"/"medium"/"high") on mispronounced_words and MTI pattern entries specifically -- keep those, they are structurally required downstream and nowhere else.

You are inventing what a deterministic speech-analysis pipeline would have MEASURED from a short recording, for a candidate with this speaking behaviour:

{trait_lines}

Write a short (15-35 word) plausible spoken-English sentence or two this candidate might say (any everyday topic — do not mention job roles, seniority, or interview difficulty), then invent Level 1 and Level 2 data that is INTERNALLY CONSISTENT with both that sentence and the speaking behaviour above.

Return ONLY a single JSON object with exactly two top-level keys, "level1" and "level2", matching these schemas exactly (same field names, same nesting, no extra top-level keys, no additional fields beyond what's listed):

LEVEL 1 (raw timeline):
{level1_schema}

LEVEL 2 (five analyzer blocks — raw measurements/events only, no scores or labels):
{level2_schema}

Hard requirements:
- "words" must be split so the sentence(s) are reconstructable; timestamps must be increasing and non-overlapping.
- "detected_phonemes": at most {max_phonemes_per_word} entries per word (a sparse sample, not a dense stream), IPA symbols wrapped in slashes, e.g. "/θ/".
- "acoustic_contours": one point roughly every {contour_hop} seconds for both pitch_hz and energy_rms, on the same time grid, spanning the recording's duration.
- Raw numbers must agree with each other and with Level 1: words_per_minute must match the actual word count and duration; fillers_per_minute must match filler_count and how many filler-like words (um, uh, like, you know) actually appear in "words" and in fluency.fillers.fillers; mispronounced_words/pattern entries must reference words that actually appear in "words" at matching timestamps; if monotone_sections cover most of the recording, pitch_range must be correspondingly low; high phoneme_accuracy must NOT co-occur with many mispronounced_words.
- stress_accuracy is always exactly 1.0 and stress_errors/stress_transfer are always empty lists — these are a permanent placeholder in the real pipeline, not something you vary.
- "engagement" is always an empty object {{}} -- production's engagement is entirely a derived synthesis of the other four blocks, with no raw measurements of its own.
- Do not add a score, a rating, a classification label, an observation string, a strength, a weakness, a recommendation, or any field not in the schemas above. "severity" on mispronounced_words/MTI patterns is the one exception -- it belongs there, as shown in the schemas.
- Return ONLY the JSON object. No markdown fences, no prose before or after it.
"""


class PromptBuilder:
    """Builds the evidence-generation prompt for one blueprint."""

    def __init__(self, config: SyntheticGenerationConfig) -> None:
        self._config = config

    def build(self, blueprint: SpeakerBlueprint) -> str:
        """Render `blueprint` into the finished prompt string.

        Raises:
            PromptBuildError: `blueprint` names a dimension/level this
                builder has no trait description for.
        """
        trait_lines = []
        for dimension, level in blueprint.as_dict().items():
            descriptions = _TRAIT_DESCRIPTIONS.get(dimension)
            description = descriptions.get(level) if descriptions else None
            if description is None:
                raise PromptBuildError(
                    f"No trait description for {dimension}={level!r} (blueprint {blueprint.blueprint_id})."
                )
            trait_lines.append(f"- {dimension.replace('_', ' ')}: {level} — {description}")

        prompt = _PROMPT_TEMPLATE.format(
            trait_lines="\n".join(trait_lines),
            level1_schema=_LEVEL1_SCHEMA,
            level2_schema=_LEVEL2_SCHEMA,
            max_phonemes_per_word=self._config.max_detected_phonemes_per_word,
            contour_hop=self._config.contour_hop_seconds,
        )
        logger.debug("Built prompt for blueprint %s (%d chars).", blueprint.blueprint_id, len(prompt))
        return prompt
