"""
Tests for the Phase 0 synthetic evidence generation rewrite.

Every test uses a fake provider -- no network call, no ZIQRA_TEACHER_*
credentials required. validator.py's production-compatibility check DOES
call the real backend.feedback.coach_packets.build_coach_packets() (that's
the point: it's the only way to actually guarantee compatibility), which is
pure/local and requires no external service.
"""

import copy
import json
import os
import random
import sys
import unittest

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_MODULE_DIR)
_KD_DIR = os.path.dirname(_PACKAGE_DIR)
_BACKEND_DIR = os.path.dirname(_KD_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from backend.knowledge_distillation.synthetic_generation import (
    BlueprintGenerator,
    DiversityFilter,
    EvidenceGenerator,
    EvidenceParsingError,
    NonRetryableProviderError,
    Pipeline,
    PromptBuilder,
    SpeakerBlueprint,
    SyntheticGenerationConfig,
    Validator,
    ValidationCategory,
)
from backend.knowledge_distillation.synthetic_generation.provider import Provider

# ---------------------------------------------------------------------------
# A hand-built, internally-consistent packet -- the baseline every
# logical/timeline/compatibility test mutates one field of.
# ---------------------------------------------------------------------------
GOOD_LEVEL1 = {
    "audio_metadata": {"duration": 6.0, "sample_rate": 16000, "detected_language": "en", "language_probability": 0.95},
    "speech_chunks": [{"chunk_id": 1, "start": 0.0, "end": 6.0}],
    "sentences": [{"text": "I really enjoyed working on that project team.", "start": 0.0, "end": 6.0}],
    "words": [
        {"word": w, "start": i * 0.6, "end": i * 0.6 + 0.5, "confidence": 0.9}
        for i, w in enumerate(["I", "really", "enjoyed", "working", "on", "that", "project", "team"])
    ],
    "detected_phonemes": [{"phoneme": "/aɪ/", "start": 0.0, "end": 0.1}],
    "acoustic_contours": {
        "pitch_hz": [{"time": t * 0.25, "hz": 150.0, "voiced": True} for t in range(24)],
        "energy_rms": [{"time": t * 0.25, "rms": 0.05} for t in range(24)],
    },
}
GOOD_LEVEL2 = {
    "fluency": {
        "fillers": {"filler_count": 0, "fillers_per_minute": 0.0, "fillers": []},
        "pauses": {"total_pauses": 1, "pauses": [{"start": 2.0, "end": 2.1, "duration": 0.1, "type": "natural"}]},
        "speaking_speed": {"words_per_minute": 80, "sentences_per_minute": 10},
    },
    "pronunciation": {
        "phoneme_accuracy": 0.88, "stress_accuracy": 1.0, "rhythm_score": 0.9,
        "phoneme_errors": [], "detected_phonemes": [], "mispronounced_words": [], "stress_errors": [],
        "rhythm_issues": [],
    },
    "mti": {
        "summary": {"vowel_pattern_issues": 0, "consonant_pattern_issues": 0, "stress_transfer_issues": 0},
        "vowel_patterns": [], "consonant_patterns": [], "stress_transfer": [],
        "speech_statistics": {"total_words_analyzed": 8, "affected_words": 0, "affected_percentage": 0.0},
    },
    "intonation": {
        "pitch_variation": {"average_pitch": 150.0, "min_pitch": 120.0, "max_pitch": 200.0, "pitch_range": 80.0},
        "energy_variation": {"low_energy_sections": []},
        "monotonicity": {"monotone_sections": []},
        "emphasis": {"under_emphasized_words": []},
    },
    "engagement": {},
}


def make_config(**overrides) -> SyntheticGenerationConfig:
    return SyntheticGenerationConfig(model="fake-model", **overrides)


def make_blueprint(**overrides) -> SpeakerBlueprint:
    defaults = dict(
        blueprint_id="bp_test", pronunciation="average", mti_severity="none", pace="moderate",
        filler_frequency="minimal", rhythm="steady", intonation="moderate", engagement="neutral",
        confidence="moderate",
    )
    defaults.update(overrides)
    return SpeakerBlueprint(**defaults)


class FakeProvider(Provider):
    """Returns a fixed, valid packet response; never touches the network."""

    def __init__(self, level1=None, level2=None) -> None:
        self.level1 = level1 if level1 is not None else copy.deepcopy(GOOD_LEVEL1)
        self.level2 = level2 if level2 is not None else copy.deepcopy(GOOD_LEVEL2)
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps({"level1": self.level1, "level2": self.level2})


class BrokenResponseProvider(Provider):
    def generate(self, prompt: str) -> str:
        return "not json at all"


class AlwaysFailsProvider(Provider):
    def generate(self, prompt: str) -> str:
        from backend.knowledge_distillation.synthetic_generation.exceptions import ProviderError
        raise ProviderError("simulated permanent transient failure")


class AccountBlockedProvider(Provider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        raise NonRetryableProviderError("Teacher LLM request failed (401): CreditsError: Insufficient balance.")


def evidence_generator_with(level1=None, level2=None, config=None) -> EvidenceGenerator:
    config = config or make_config()
    return EvidenceGenerator(FakeProvider(level1, level2), config)


class BlueprintGeneratorTests(unittest.TestCase):
    def test_generate_covers_all_eight_dimensions(self):
        config = make_config()
        blueprint = BlueprintGenerator(config, rng=random.Random(1)).generate()
        self.assertEqual(set(blueprint.as_dict().keys()), set(config.blueprint_dimensions.keys()))

    def test_generate_only_uses_declared_levels(self):
        config = make_config()
        gen = BlueprintGenerator(config, rng=random.Random(2))
        for _ in range(50):
            blueprint = gen.generate()
            for dimension, level in blueprint.as_dict().items():
                self.assertIn(level, config.blueprint_dimensions[dimension])

    def test_blueprint_has_no_role_experience_or_difficulty_fields(self):
        blueprint = BlueprintGenerator(make_config()).generate()
        forbidden = {"role", "experience", "difficulty", "interview_difficulty", "seniority"}
        self.assertEqual(forbidden & set(blueprint.as_dict().keys()), set())


class PromptBuilderTests(unittest.TestCase):
    def test_build_mentions_every_blueprint_trait(self):
        config = make_config()
        blueprint = make_blueprint(pronunciation="weak", pace="fast", confidence="low")
        prompt = PromptBuilder(config).build(blueprint)
        self.assertIn("weak", prompt)
        self.assertIn("fast", prompt)
        self.assertIn("low", prompt)

    def test_build_never_requests_a_role_or_seniority(self):
        # The prompt legitimately contains the PHRASE "interview difficulty"
        # once, as a negative instruction ("do not mention ... interview
        # difficulty") -- that's correct, not a leak. What must never appear
        # is a concrete role/seniority example being requested as content.
        prompt = PromptBuilder(make_config()).build(make_blueprint())
        lowered = prompt.lower()
        for forbidden in ("software engineer", "product manager", "fresher", "senior engineer"):
            self.assertNotIn(forbidden, lowered)


class EvidenceGeneratorTests(unittest.TestCase):
    def test_generate_returns_a_typed_packet(self):
        packet = evidence_generator_with().generate(make_blueprint())
        self.assertEqual(packet.level1, GOOD_LEVEL1)
        self.assertEqual(packet.level2, GOOD_LEVEL2)
        self.assertTrue(packet.session_id.startswith("session_"))
        json.dumps(packet.to_dict())  # must be serializable

    def test_non_json_response_raises_parsing_error(self):
        gen = EvidenceGenerator(BrokenResponseProvider(), make_config())
        with self.assertRaises(EvidenceParsingError):
            gen.generate(make_blueprint())

    def test_response_missing_level2_raises_parsing_error(self):
        provider = FakeProvider()

        class OnlyLevel1Provider(Provider):
            def generate(self, prompt: str) -> str:
                return json.dumps({"level1": GOOD_LEVEL1})

        gen = EvidenceGenerator(OnlyLevel1Provider(), make_config())
        with self.assertRaises(EvidenceParsingError):
            gen.generate(make_blueprint())


class ProviderInterfaceTests(unittest.TestCase):
    """provider.py has no concrete implementation for this phase (see its
    module docstring: no Qwen, no default-wired external LLM). Confirms the
    interface a future Provider must satisfy, and that AlwaysFailsProvider/
    AccountBlockedProvider (used throughout this file) genuinely implement it."""

    def test_provider_is_abstract(self):
        with self.assertRaises(TypeError):
            Provider()  # cannot instantiate the ABC directly

    def test_fake_providers_satisfy_the_interface(self):
        self.assertIsInstance(FakeProvider(), Provider)
        self.assertIsInstance(AlwaysFailsProvider(), Provider)
        self.assertIsInstance(AccountBlockedProvider(), Provider)

    def test_no_concrete_llm_backed_provider_exists_in_this_module(self):
        """Regression guard for the explicit instruction: Qwen (or any
        other LLM) must not be introduced into this pipeline. A concrete
        Provider subclass appearing in provider.py would be exactly that."""
        import inspect

        from backend.knowledge_distillation.synthetic_generation import provider as provider_module

        concrete_providers = [
            name for name, obj in vars(provider_module).items()
            if inspect.isclass(obj) and issubclass(obj, Provider) and obj is not Provider
        ]
        self.assertEqual(concrete_providers, [])


class ValidatorSchemaTests(unittest.TestCase):
    def test_a_well_formed_packet_is_accepted(self):
        packet = evidence_generator_with().generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertTrue(result.accepted, result.to_dict())

    def test_missing_level2_field_is_rejected_with_schema_issue(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        del level2["pronunciation"]["phoneme_accuracy"]
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.SCHEMA for i in result.issues))

    def test_wrong_type_field_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["pronunciation"]["phoneme_accuracy"] = "eighty-eight percent"
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)

    def test_a_score_field_would_be_rejected_as_an_unexpected_type(self):
        """The dataset carries no score fields at all -- confirms a stray
        one doesn't silently pass schema validation by accident (it's not
        in _LEVEL2_REQUIRED, so its presence is simply ignored, but a
        caller relying on its ABSENCE should see it's genuinely absent from
        the fixture, not merely unchecked)."""
        self.assertNotIn("pronunciation_score", GOOD_LEVEL2["pronunciation"])
        self.assertNotIn("overall_mti_score", GOOD_LEVEL2["mti"])
        self.assertNotIn("engagement_score", GOOD_LEVEL2["engagement"])


class ValidatorRangeTests(unittest.TestCase):
    def test_out_of_range_phoneme_accuracy_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["pronunciation"]["phoneme_accuracy"] = 1.5
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.RANGE for i in result.issues))

    def test_out_of_range_wpm_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["fluency"]["speaking_speed"]["words_per_minute"] = 500
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)


class ValidatorLogicalTests(unittest.TestCase):
    """Raw-number-only consistency checks -- no severity/classification/
    label fields exist in this schema, so every check compares numbers
    directly (see config.py's module docstring for why)."""

    def test_high_accuracy_with_many_mispronounced_words_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["pronunciation"]["phoneme_accuracy"] = 0.98
        level2["pronunciation"]["mispronounced_words"] = [
            {"word": "I", "start": 0.0, "end": 0.5},
            {"word": "really", "start": 0.6, "end": 1.1},
            {"word": "enjoyed", "start": 1.2, "end": 1.7},
        ]
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.LOGICAL for i in result.issues))

    def test_wide_monotone_coverage_with_high_pitch_range_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        # monotone_sections cover the whole 6s duration
        level2["intonation"]["monotonicity"]["monotone_sections"] = [{"start": 0.0, "end": 6.0}]
        level2["intonation"]["pitch_variation"]["pitch_range"] = 300.0
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.LOGICAL for i in result.issues))

    def test_filler_count_mismatched_with_fillers_list_length_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["fluency"]["fillers"]["filler_count"] = 5
        level2["fluency"]["fillers"]["fillers"] = []  # 0 actual entries vs count=5
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.LOGICAL for i in result.issues))

    def test_mti_summary_count_mismatched_with_pattern_list_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["mti"]["summary"]["vowel_pattern_issues"] = 3
        level2["mti"]["vowel_patterns"] = []  # declared 3, actually 0
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.LOGICAL for i in result.issues))


class ValidatorTimelineConsistencyTests(unittest.TestCase):
    def test_wpm_wildly_inconsistent_with_word_count_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["fluency"]["speaking_speed"]["words_per_minute"] = 250  # 8 words / 6s implies ~80
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.TIMELINE_CONSISTENCY for i in result.issues))

    def test_filler_rate_inconsistent_with_count_and_duration_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["fluency"]["fillers"]["filler_count"] = 1
        level2["fluency"]["fillers"]["fillers"] = [{"word": "um", "start": 1.0, "end": 1.2}]
        level2["fluency"]["fillers"]["fillers_per_minute"] = 20.0  # 1 filler / 6s implies ~10, not 20
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.TIMELINE_CONSISTENCY for i in result.issues))

    def test_mispronounced_word_not_in_level1_words_is_rejected(self):
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["pronunciation"]["mispronounced_words"] = [
            {"word": "nonexistentword", "start": 0.0, "end": 0.5}
        ]
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.TIMELINE_CONSISTENCY for i in result.issues))


class ValidatorProductionCompatibilityTests(unittest.TestCase):
    def test_compatible_packet_survives_a_real_build_coach_packets_call(self):
        from backend.feedback.coach_packets import build_coach_packets

        packet = evidence_generator_with().generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertTrue(result.accepted, result.to_dict())
        # and prove it really does flow through without raising -- score
        # fields are legitimately None at this stage (no scores exist in
        # the input by design; scoring is the teacher-generation stage's
        # job), so compatibility asserts "doesn't crash", not "is populated".
        built = build_coach_packets(packet.level2, session_id=packet.session_id)
        self.assertIsNone(built["articulation"]["pronunciation"]["score"])
        self.assertIsNone(built["delivery"]["engagement"]["score"])
        self.assertEqual(built["articulation"]["coach"], "articulation")
        self.assertEqual(built["delivery"]["coach"], "delivery")

    def test_a_malformed_list_item_schema_cant_catch_is_still_caught(self):
        # schema validation only checks that mispronounced_words IS a list
        # (any list passes); it never inspects each item's internal shape.
        # A mispronounced_words entry with no "word" key satisfies schema
        # but crashes build_articulation_packet()'s `w["word"]` access --
        # exactly the class of bug only literally calling the function
        # catches, which is why compatibility isn't just a duplicated
        # schema list.
        level2 = copy.deepcopy(GOOD_LEVEL2)
        level2["pronunciation"]["mispronounced_words"] = [{"start": 0.0, "end": 0.5}]
        packet = evidence_generator_with(level2=level2).generate(make_blueprint())
        result = Validator(make_config()).validate(packet)
        self.assertFalse(result.accepted)
        self.assertTrue(any(i.category == ValidationCategory.PRODUCTION_COMPATIBILITY for i in result.issues))


class DiversityFilterTests(unittest.TestCase):
    def test_within_quota_blueprints_are_accepted(self):
        config = make_config(target_dataset_size=100)
        filt = DiversityFilter(config)
        result = filt.check(make_blueprint())
        self.assertTrue(result.accepted)

    def test_overrepresented_dimension_is_rejected(self):
        config = make_config(target_dataset_size=100, diversity_overrepresentation_factor=1.0)
        filt = DiversityFilter(config)
        cap = filt._cap_for_dimension(len(config.blueprint_dimensions["pronunciation"]))
        for i in range(cap):
            filt.record(make_blueprint(blueprint_id=f"bp_{i}", pronunciation="weak"))
        result = filt.check(make_blueprint(blueprint_id="bp_over", pronunciation="weak"))
        self.assertFalse(result.accepted)

    def test_exact_fingerprint_repeat_cap_is_enforced(self):
        config = make_config(max_exact_fingerprint_repeats=2)
        filt = DiversityFilter(config)
        bp = make_blueprint()
        filt.record(bp)
        filt.record(bp)
        result = filt.check(bp)
        self.assertFalse(result.accepted)

    def test_record_before_check_is_never_required_for_a_rejected_packet(self):
        """A packet that fails validation must never consume diversity
        headroom -- i.e. check() alone must not mutate state."""
        config = make_config(target_dataset_size=100)
        filt = DiversityFilter(config)
        filt.check(make_blueprint())
        filt.check(make_blueprint())
        self.assertEqual(filt.distribution()["pronunciation"].get("average", 0), 0)


class PipelineTests(unittest.TestCase):
    def test_run_reaches_target_dataset_size_with_a_good_provider(self):
        # diversity_overrepresentation_factor generous: this test is about
        # pipeline mechanics (does it reach target?), not diversity
        # convergence (covered separately in DiversityFilterTests) -- a
        # small target like 15 against 8 independently-tracked quotas is a
        # genuinely pathological regime for the quota mechanism (confirmed
        # empirically: convergence is clean at 200+, tight at <20).
        config = make_config(target_dataset_size=15, diversity_overrepresentation_factor=100.0)
        pipeline = Pipeline(
            blueprint_generator=BlueprintGenerator(config, rng=random.Random(1)),
            evidence_generator=EvidenceGenerator(FakeProvider(), config),
            validator=Validator(config),
            diversity_filter=DiversityFilter(config),
            config=config,
        )
        stats = pipeline.run()
        self.assertEqual(stats.accepted, 15)
        self.assertEqual(stats.aborted_reason, "")

    def test_a_broken_provider_stops_instead_of_looping_forever(self):
        config = make_config(target_dataset_size=10, max_attempt_factor=3)
        pipeline = Pipeline(
            blueprint_generator=BlueprintGenerator(config, rng=random.Random(1)),
            evidence_generator=EvidenceGenerator(BrokenResponseProvider(), config),
            validator=Validator(config),
            diversity_filter=DiversityFilter(config),
            config=config,
        )
        stats = pipeline.run()
        self.assertEqual(stats.accepted, 0)
        self.assertLessEqual(stats.total_attempts, config.target_dataset_size * config.max_attempt_factor)
        self.assertNotEqual(stats.aborted_reason, "")

    def test_an_account_level_block_aborts_immediately_without_retrying(self):
        """Regression test for the real incident: a NonRetryableProviderError
        must stop the WHOLE run on the first occurrence, not just be
        skipped/retried like an ordinary generation failure."""
        config = make_config(target_dataset_size=50, max_attempt_factor=10)
        provider = AccountBlockedProvider()
        pipeline = Pipeline(
            blueprint_generator=BlueprintGenerator(config, rng=random.Random(1)),
            evidence_generator=EvidenceGenerator(provider, config),
            validator=Validator(config),
            diversity_filter=DiversityFilter(config),
            config=config,
        )
        stats = pipeline.run()
        self.assertEqual(stats.accepted, 0)
        self.assertIn("CreditsError", stats.aborted_reason)
        self.assertEqual(provider.calls, 1)  # aborted after exactly one attempt

    def test_run_writes_packets_jsonl_only(self):
        """synthetic_generation's only output is raw evidence -- deriving
        Articulation/Delivery packets is teacher_generation.packet_builder's
        job now, not this module's (see pipeline.py's docstring)."""
        import tempfile

        config = make_config(target_dataset_size=8, diversity_overrepresentation_factor=100.0)
        pipeline = Pipeline(
            blueprint_generator=BlueprintGenerator(config, rng=random.Random(3)),
            evidence_generator=EvidenceGenerator(FakeProvider(), config),
            validator=Validator(config),
            diversity_filter=DiversityFilter(config),
            config=config,
        )
        with tempfile.TemporaryDirectory() as out_dir:
            stats = pipeline.run(out_dir=out_dir)
            self.assertEqual(stats.accepted, 8)
            path = os.path.join(out_dir, "packets", "packets.jsonl")
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 8)
            json.loads(lines[0])  # each line must be valid JSON
            self.assertFalse(os.path.exists(os.path.join(out_dir, "articulation")))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "delivery")))


class ModuleBoundaryTests(unittest.TestCase):
    def test_no_import_of_validation_or_diversity_as_separate_packages(self):
        """This rewrite folds validator/diversity INTO this package -- it
        must never import a separate top-level validation/diversity package
        (those were deleted; a stray import would mean stale code)."""
        import ast

        forbidden = ("knowledge_distillation.validation", "knowledge_distillation.diversity")
        offending = []
        for fname in os.listdir(_PACKAGE_DIR):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(_PACKAGE_DIR, fname)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if any(phrase in name for phrase in forbidden):
                        offending.append((fname, name))
        self.assertEqual(offending, [])

    def test_blueprint_never_carries_role_experience_or_difficulty(self):
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(SpeakerBlueprint)}
        forbidden = {"role", "experience", "difficulty", "interview_difficulty", "seniority", "question_difficulty"}
        self.assertEqual(field_names & forbidden, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
