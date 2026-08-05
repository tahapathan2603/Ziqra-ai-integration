"""
Validates one SyntheticEvidencePacket against five independent checks:

  1. schema                    -- required fields present, correct types
  2. range                     -- numeric values within valid bounds
  3. logical                   -- raw numbers/counts must agree with each other
  4. timeline_consistency      -- Level 1 and Level 2 agree with each other
  5. production_compatibility  -- build_coach_packets() actually runs on it

A packet is ACCEPTED only if every check passes. Every issue is collected
(not just the first), so a rejected packet's full report is inspectable —
useful when tuning authored content rather than debugging one at a time.

RAW EVIDENCE ONLY (see config.py's module docstring): the dataset carries no
scores, ratings, or classification labels, so every check here compares raw
numbers to each other directly -- never a label to a metric. ONE exception:
mispronounced_words[].severity and MTI patterns[].severity are required,
because build_coach_packets() reads them via hard bracket indexing
(`w["severity"]`) and crashes without them -- confirmed empirically, not
assumed (see prompt_builder.py's docstring for the full explanation).

Compatibility (5) is checked by literally calling the real
backend.feedback.coach_packets.build_coach_packets() and confirming it
doesn't raise, rather than asserting its output is fully populated: with no
scores in the input, build_coach_packets()'s own score/level fields are
legitimately None at this stage (scoring is the teacher-generation stage's
job -- see [[project_llm_assigns_scores_not_pipeline]]) -- that's expected,
not a compatibility failure. Only a crash is.
"""

import logging
from typing import Any, List, Tuple

from ...feedback.coach_packets import build_coach_packets
from .config import SyntheticGenerationConfig
from .schemas import SyntheticEvidencePacket, ValidationCategory, ValidationIssue, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

_MISSING = object()


def _get_path(container: Any, path: str) -> Any:
    """Resolve a dotted path (e.g. "pronunciation.phoneme_accuracy") through
    nested dicts. Returns _MISSING if any segment is absent or the
    container along the way isn't a dict."""
    node = container
    for segment in path.split("."):
        if not isinstance(node, dict) or segment not in node:
            return _MISSING
        node = node[segment]
    return node


def _issue(category: ValidationCategory, field: str, message: str) -> ValidationIssue:
    return ValidationIssue(category=category, field=field, message=message)


class Validator:
    """Runs all five checks over one packet and returns a ValidationResult."""

    def __init__(self, config: SyntheticGenerationConfig) -> None:
        self._config = config

    def validate(self, packet: SyntheticEvidencePacket) -> ValidationResult:
        issues: List[ValidationIssue] = []
        issues += self._check_schema(packet)

        # Range/logical/timeline checks assume the fields schema validation
        # requires actually exist -- skip them on a schema failure rather
        # than raising KeyError trying to read a field that isn't there.
        if not issues:
            issues += self._check_ranges(packet)
            issues += self._check_logical_consistency(packet)
            issues += self._check_timeline_consistency(packet)
            issues += self._check_production_compatibility(packet)

        status = ValidationStatus.REJECTED if issues else ValidationStatus.ACCEPTED
        return ValidationResult(status=status, issues=issues)

    # -- 1. schema ---------------------------------------------------------
    _LEVEL1_REQUIRED: Tuple[Tuple[str, Tuple[type, ...]], ...] = (
        ("audio_metadata.duration", (int, float)),
        ("audio_metadata.sample_rate", (int,)),
        ("audio_metadata.detected_language", (str,)),
        ("audio_metadata.language_probability", (int, float)),
        ("speech_chunks", (list,)),
        ("sentences", (list,)),
        ("words", (list,)),
        ("detected_phonemes", (list,)),
        ("acoustic_contours.pitch_hz", (list,)),
        ("acoustic_contours.energy_rms", (list,)),
    )
    _LEVEL2_REQUIRED: Tuple[Tuple[str, Tuple[type, ...]], ...] = (
        ("fluency.fillers.filler_count", (int,)),
        ("fluency.fillers.fillers_per_minute", (int, float)),
        ("fluency.fillers.fillers", (list,)),
        ("fluency.pauses.total_pauses", (int,)),
        ("fluency.pauses.pauses", (list,)),
        ("fluency.speaking_speed.words_per_minute", (int, float)),
        ("pronunciation.phoneme_accuracy", (int, float)),
        ("pronunciation.stress_accuracy", (int, float)),
        ("pronunciation.rhythm_score", (int, float)),
        ("pronunciation.phoneme_errors", (list,)),
        ("pronunciation.mispronounced_words", (list,)),
        ("pronunciation.stress_errors", (list,)),
        ("pronunciation.rhythm_issues", (list,)),
        ("mti.summary.vowel_pattern_issues", (int,)),
        ("mti.summary.consonant_pattern_issues", (int,)),
        ("mti.vowel_patterns", (list,)),
        ("mti.consonant_patterns", (list,)),
        ("mti.stress_transfer", (list,)),
        ("mti.speech_statistics.total_words_analyzed", (int,)),
        ("mti.speech_statistics.affected_words", (int,)),
        ("intonation.pitch_variation.average_pitch", (int, float)),
        ("intonation.pitch_variation.pitch_range", (int, float)),
        ("intonation.energy_variation.low_energy_sections", (list,)),
        ("intonation.monotonicity.monotone_sections", (list,)),
        ("intonation.emphasis.under_emphasized_words", (list,)),
        ("engagement", (dict,)),
    )

    def _check_schema(self, packet: SyntheticEvidencePacket) -> List[ValidationIssue]:
        issues = []
        for path, types in self._LEVEL1_REQUIRED:
            value = _get_path(packet.level1, path)
            if value is _MISSING:
                issues.append(_issue(ValidationCategory.SCHEMA, f"level1.{path}", "missing required field"))
            elif not isinstance(value, types):
                issues.append(
                    _issue(
                        ValidationCategory.SCHEMA, f"level1.{path}",
                        f"expected {types}, got {type(value).__name__}",
                    )
                )
        for path, types in self._LEVEL2_REQUIRED:
            value = _get_path(packet.level2, path)
            if value is _MISSING:
                issues.append(_issue(ValidationCategory.SCHEMA, f"level2.{path}", "missing required field"))
            elif not isinstance(value, types):
                issues.append(
                    _issue(
                        ValidationCategory.SCHEMA, f"level2.{path}",
                        f"expected {types}, got {type(value).__name__}",
                    )
                )
        for word in packet.level1.get("words", []) if isinstance(packet.level1.get("words"), list) else []:
            if not isinstance(word, dict) or not all(k in word for k in ("word", "start", "end", "confidence")):
                issues.append(_issue(ValidationCategory.SCHEMA, "level1.words[]", f"malformed word entry: {word!r}"))
        return issues

    # -- 2. range ------------------------------------------------------------
    def _check_ranges(self, packet: SyntheticEvidencePacket) -> List[ValidationIssue]:
        c = self._config
        issues = []

        def in_range(path: str, value: float, bounds: Tuple[float, float]) -> None:
            lo, hi = bounds
            if not (lo <= value <= hi):
                issues.append(_issue(ValidationCategory.RANGE, path, f"{value} outside [{lo}, {hi}]"))

        in_range("level2.pronunciation.phoneme_accuracy", packet.level2["pronunciation"]["phoneme_accuracy"], c.fraction_range)
        in_range("level2.pronunciation.rhythm_score", packet.level2["pronunciation"]["rhythm_score"], c.fraction_range)
        in_range("level2.fluency.speaking_speed.words_per_minute", packet.level2["fluency"]["speaking_speed"]["words_per_minute"], c.wpm_range)
        in_range("level2.fluency.fillers.fillers_per_minute", packet.level2["fluency"]["fillers"]["fillers_per_minute"], c.fillers_per_minute_range)
        in_range("level1.audio_metadata.duration", packet.level1["audio_metadata"]["duration"], c.duration_range_seconds)

        for i, word in enumerate(packet.level1.get("words", [])):
            if not isinstance(word, dict):
                continue
            confidence = word.get("confidence")
            if isinstance(confidence, (int, float)) and not (c.fraction_range[0] <= confidence <= c.fraction_range[1]):
                issues.append(_issue(ValidationCategory.RANGE, f"level1.words[{i}].confidence", f"{confidence} outside {c.fraction_range}"))
            start, end = word.get("start"), word.get("end")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start >= end:
                issues.append(_issue(ValidationCategory.RANGE, f"level1.words[{i}]", f"start ({start}) not before end ({end})"))

        return issues

    # -- 3. logical consistency (raw numbers vs raw numbers only) ------------
    def _check_logical_consistency(self, packet: SyntheticEvidencePacket) -> List[ValidationIssue]:
        c = self._config
        issues = []
        pronunciation = packet.level2["pronunciation"]
        fluency = packet.level2["fluency"]
        intonation = packet.level2["intonation"]
        duration = packet.level1["audio_metadata"]["duration"]

        # High accuracy with a large raw count of mispronounced words. No
        # severity to weigh (none exists in this schema) -- just volume.
        accuracy = pronunciation["phoneme_accuracy"]
        mispronounced_count = len(pronunciation["mispronounced_words"])
        if accuracy >= c.high_accuracy_threshold and mispronounced_count > c.max_mispronounced_words_at_high_accuracy:
            issues.append(
                _issue(
                    ValidationCategory.LOGICAL, "pronunciation",
                    f"phoneme_accuracy={accuracy} is high but {mispronounced_count} mispronounced_words present "
                    f"(max {c.max_mispronounced_words_at_high_accuracy})",
                )
            )

        # Monotone-section coverage vs pitch_range: if most of the recording
        # is inside a monotone section, pitch_range can't also be wide.
        monotone_sections = intonation["monotonicity"]["monotone_sections"]
        if monotone_sections and duration > 0:
            covered = sum(
                max(0.0, min(s.get("end", 0), duration) - max(s.get("start", 0), 0.0))
                for s in monotone_sections if isinstance(s, dict)
            )
            coverage_ratio = covered / duration
            pitch_range = intonation["pitch_variation"]["pitch_range"]
            if coverage_ratio >= c.monotone_coverage_ratio_for_flat_check and pitch_range > c.flat_max_pitch_range_hz:
                issues.append(
                    _issue(
                        ValidationCategory.LOGICAL, "intonation",
                        f"monotone_sections cover {coverage_ratio:.0%} of the recording but pitch_range={pitch_range} "
                        f"> {c.flat_max_pitch_range_hz}",
                    )
                )

        # fillers[] list length should roughly match filler_count.
        filler_count = fluency["fillers"]["filler_count"]
        filler_list_length = len(fluency["fillers"]["fillers"])
        if abs(filler_list_length - filler_count) > c.filler_list_length_tolerance:
            issues.append(
                _issue(
                    ValidationCategory.LOGICAL, "fluency.fillers",
                    f"filler_count={filler_count} but fillers[] has {filler_list_length} entries",
                )
            )

        # mti.summary counts should match the actual pattern list lengths.
        mti = packet.level2["mti"]
        vowel_declared = mti["summary"]["vowel_pattern_issues"]
        vowel_actual = len(mti["vowel_patterns"])
        if vowel_declared != vowel_actual:
            issues.append(
                _issue(
                    ValidationCategory.LOGICAL, "mti.summary.vowel_pattern_issues",
                    f"declared {vowel_declared} but vowel_patterns has {vowel_actual} entries",
                )
            )
        consonant_declared = mti["summary"]["consonant_pattern_issues"]
        consonant_actual = len(mti["consonant_patterns"])
        if consonant_declared != consonant_actual:
            issues.append(
                _issue(
                    ValidationCategory.LOGICAL, "mti.summary.consonant_pattern_issues",
                    f"declared {consonant_declared} but consonant_patterns has {consonant_actual} entries",
                )
            )

        return issues

    # -- 4. timeline consistency (Level 1 vs Level 2) -------------------------
    def _check_timeline_consistency(self, packet: SyntheticEvidencePacket) -> List[ValidationIssue]:
        c = self._config
        issues = []
        words = [w for w in packet.level1.get("words", []) if isinstance(w, dict)]
        duration = packet.level1["audio_metadata"]["duration"]

        # wpm computed from the actual word count/duration should roughly
        # match the reported words_per_minute.
        if words and duration > 0:
            computed_wpm = len(words) / (duration / 60.0)
            reported_wpm = packet.level2["fluency"]["speaking_speed"]["words_per_minute"]
            if reported_wpm > 0:
                relative_error = abs(computed_wpm - reported_wpm) / reported_wpm
                if relative_error > c.wpm_cross_check_tolerance:
                    issues.append(
                        _issue(
                            ValidationCategory.TIMELINE_CONSISTENCY, "fluency.speaking_speed.words_per_minute",
                            f"reported wpm={reported_wpm} but {len(words)} words over {duration}s implies ~{computed_wpm:.0f}",
                        )
                    )

        # fillers_per_minute should match filler_count and duration.
        filler_count = packet.level2["fluency"]["fillers"]["filler_count"]
        fillers_per_minute = packet.level2["fluency"]["fillers"]["fillers_per_minute"]
        if duration > 0:
            computed_rate = filler_count / (duration / 60.0)
            if computed_rate > 0:
                relative_error = abs(computed_rate - fillers_per_minute) / computed_rate
                if relative_error > c.filler_rate_cross_check_tolerance:
                    issues.append(
                        _issue(
                            ValidationCategory.TIMELINE_CONSISTENCY, "fluency.fillers.fillers_per_minute",
                            f"reported fillers_per_minute={fillers_per_minute} but filler_count={filler_count} over "
                            f"{duration}s implies ~{computed_rate:.1f}",
                        )
                    )
            elif fillers_per_minute > 0:
                issues.append(
                    _issue(
                        ValidationCategory.TIMELINE_CONSISTENCY, "fluency.fillers.fillers_per_minute",
                        f"filler_count=0 but fillers_per_minute={fillers_per_minute}",
                    )
                )

        # Words cited in Level 2 (mispronounced/patterns) should actually
        # appear in Level 1's word list.
        level1_words = {w["word"].lower() for w in words if isinstance(w.get("word"), str)}
        cited_words = []
        cited_words += [w.get("word") for w in packet.level2["pronunciation"]["mispronounced_words"]]
        cited_words += [p.get("word") for p in packet.level2["mti"]["vowel_patterns"]]
        cited_words += [p.get("word") for p in packet.level2["mti"]["consonant_patterns"]]
        cited_words = [w for w in cited_words if isinstance(w, str)]
        if cited_words:
            matched = sum(1 for w in cited_words if w.lower() in level1_words)
            ratio = matched / len(cited_words)
            if ratio < c.min_word_match_ratio:
                issues.append(
                    _issue(
                        ValidationCategory.TIMELINE_CONSISTENCY, "level2 cited words",
                        f"only {matched}/{len(cited_words)} words cited in Level 2 appear in level1.words",
                    )
                )

        return issues

    # -- 5. production compatibility ------------------------------------------
    def _check_production_compatibility(self, packet: SyntheticEvidencePacket) -> List[ValidationIssue]:
        """Confirms build_coach_packets() runs on this packet without
        raising. It does NOT assert score/level fields are populated --
        with no scores in the input (by design, see module docstring),
        those fields are legitimately None in the output at this stage."""
        try:
            build_coach_packets(packet.level2, session_id=packet.session_id)
        except Exception as e:  # noqa: BLE001 -- any failure here IS the finding
            return [
                _issue(
                    ValidationCategory.PRODUCTION_COMPATIBILITY, "build_coach_packets",
                    f"build_coach_packets() raised {type(e).__name__}: {e}",
                )
            ]
        return []
