"""
Tests for teacher_generation.claude_teacher. All local/pure -- no network
call, no LLM SDK involved anywhere (that's the entire point of this
module).
"""

import json
import os
import sys
import unittest

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_MODULE_DIR)
_KD_DIR = os.path.dirname(_PACKAGE_DIR)
_BACKEND_DIR = os.path.dirname(_KD_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from backend.knowledge_distillation.teacher_generation.claude_teacher import (
    ClaudeTeacherProvider,
    _extract_session_and_evidence,
    synthesize_articulation,
    synthesize_delivery,
)
from backend.knowledge_distillation.teacher_generation.prompt_builder import (
    build_articulation_prompt,
    build_delivery_prompt,
)

CLEAN_ARTICULATION_PACKET = {
    "coach": "articulation",
    "pronunciation": {"score": None, "phoneme_accuracy_pct": 98, "observations": [], "mispronounced_words": []},
    "mti": {"score": None, "clarity_impact": None, "top_recurring_issue": None, "patterns_detected": [], "clarity_risk_words": [], "observations": [], "patterns": []},
}

ISSUE_ARTICULATION_PACKET = {
    "coach": "articulation",
    "pronunciation": {
        "score": None, "phoneme_accuracy_pct": 60, "observations": [],
        "mispronounced_words": [
            {"word": "call", "severity": "high", "start": 0.76, "end": 0.98},
            {"word": "thick", "severity": "high", "start": 6.22, "end": 6.56},
            {"word": "will", "severity": "medium", "start": 17.66, "end": 17.84},
        ],
    },
    "mti": {
        "score": None, "clarity_impact": None, "top_recurring_issue": None, "patterns_detected": [], "clarity_risk_words": [],
        "observations": [],
        "patterns": [
            {"word": "will", "issue": "consonant_substitution", "expected_phoneme": "/w/", "detected_phoneme": "/v/", "severity": "high", "start": 17.66, "end": 17.84},
            {"word": "please", "issue": "consonant_substitution", "expected_phoneme": "/z/", "detected_phoneme": "/s/", "severity": "medium", "start": 0.6, "end": 0.76},
        ],
    },
}

CLEAN_DELIVERY_PACKET = {
    "coach": "delivery",
    "fluency": {
        "speaking_speed": {"words_per_minute": 130, "classification": None},
        "filler_usage": {"count": 0, "per_minute": 0.0, "classification": None, "examples": []},
        "pauses": {"total": 0, "dead_air_count": 0, "long_pause_count": 0, "hesitation_count": 0},
    },
    "intonation": {"score": None, "delivery_label": None, "observations": [], "pitch": {"average_hz": 140.0, "range_hz": 110.0}, "flat_sections": []},
    "rhythm": {"score_pct": 92, "issues": []},
    "engagement": {"score": None, "level": None},
}

ISSUE_DELIVERY_PACKET = {
    "coach": "delivery",
    "fluency": {
        "speaking_speed": {"words_per_minute": 226, "classification": None},
        "filler_usage": {"count": 8, "per_minute": 13.1, "classification": None, "examples": ["um", "um", "uh"]},
        "pauses": {"total": 5, "dead_air_count": 1, "long_pause_count": 2, "hesitation_count": 1},
    },
    "intonation": {
        "score": None, "delivery_label": None, "observations": [],
        "pitch": {"average_hz": 110.0, "range_hz": 18.0},
        # Representative of the repaired synthetic dataset: at most one
        # monotone_section (production/repair_packets.py never emit more
        # than one) and severity restricted to {"high", "medium"} (there is
        # no "low" tier anywhere in production or the synthetic schema).
        "flat_sections": [
            {"kind": "monotone", "start": 0.6, "end": 6.6, "severity": "high"},
            {"kind": "low_energy", "start": 7.0, "end": 9.0, "severity": "high"},
            {"kind": "low_energy", "start": 10.0, "end": 11.5, "severity": "high"},
            {"kind": "low_energy", "start": 13.0, "end": 14.0, "severity": "medium"},
        ],
    },
    "rhythm": {"score_pct": 20, "issues": []},
    "engagement": {"score": None, "level": None},
}


class ExtractSessionAndEvidenceTests(unittest.TestCase):
    def test_round_trips_through_prompt_builder(self):
        prompt = build_articulation_prompt(CLEAN_ARTICULATION_PACKET, "session_abc")
        session_id, evidence = _extract_session_and_evidence(prompt)
        self.assertEqual(session_id, "session_abc")
        self.assertEqual(evidence, CLEAN_ARTICULATION_PACKET)

    def test_round_trips_for_delivery_too(self):
        prompt = build_delivery_prompt(CLEAN_DELIVERY_PACKET, "session_xyz")
        session_id, evidence = _extract_session_and_evidence(prompt)
        self.assertEqual(session_id, "session_xyz")
        self.assertEqual(evidence, CLEAN_DELIVERY_PACKET)

    def test_raises_on_unrecognized_text(self):
        with self.assertRaises(ValueError):
            _extract_session_and_evidence("not a prompt_builder prompt at all")


_SCORE_REASONING_FIELDS = {"justification", "level1_evidence", "level2_evidence", "patterns_considered", "why_not_higher", "why_not_lower"}


class SynthesizeArticulationTests(unittest.TestCase):
    def test_output_has_the_five_top_level_keys_in_order(self):
        result = synthesize_articulation(CLEAN_ARTICULATION_PACKET, "session_1")
        self.assertEqual(
            list(result.keys()),
            ["evaluation_analysis", "scores", "score_reasoning", "coach_output", "reasoning_trace"],
        )

    def test_evaluation_analysis_has_articulation_dimensions(self):
        result = synthesize_articulation(CLEAN_ARTICULATION_PACKET, "session_1")
        self.assertEqual(
            set(result["evaluation_analysis"].keys()),
            {"speech_behavior_summary", "pronunciation_quality", "mti_characteristics"},
        )
        for text in result["evaluation_analysis"].values():
            self.assertTrue(text.strip())

    def test_evaluation_analysis_varies_with_evidence(self):
        clean = synthesize_articulation(CLEAN_ARTICULATION_PACKET, "session_1")
        issue = synthesize_articulation(ISSUE_ARTICULATION_PACKET, "session_1")
        self.assertNotEqual(
            clean["evaluation_analysis"]["pronunciation_quality"], issue["evaluation_analysis"]["pronunciation_quality"]
        )
        self.assertNotEqual(
            clean["evaluation_analysis"]["mti_characteristics"], issue["evaluation_analysis"]["mti_characteristics"]
        )

    def test_score_reasoning_covers_every_rubric_with_the_right_shape(self):
        result = synthesize_articulation(ISSUE_ARTICULATION_PACKET, "session_1")
        self.assertEqual(set(result["score_reasoning"].keys()), {"pronunciation", "mti"})
        for rubric_reasoning in result["score_reasoning"].values():
            self.assertEqual(set(rubric_reasoning.keys()), _SCORE_REASONING_FIELDS)
            self.assertTrue(rubric_reasoning["justification"])
            self.assertTrue(rubric_reasoning["level1_evidence"])
            self.assertTrue(rubric_reasoning["level2_evidence"])
            self.assertTrue(rubric_reasoning["why_not_higher"])
            self.assertTrue(rubric_reasoning["why_not_lower"])

    def test_score_reasoning_ceiling_and_floor_language_matches_the_score(self):
        clean = synthesize_articulation(CLEAN_ARTICULATION_PACKET, "session_1")  # pronunciation/mti both score 5
        self.assertIn("top band", clean["score_reasoning"]["pronunciation"]["why_not_higher"])
        self.assertIn("top band", clean["score_reasoning"]["mti"]["why_not_higher"])

    def test_reasoning_trace_includes_evaluation_analysis_entries(self):
        result = synthesize_articulation(CLEAN_ARTICULATION_PACKET, "session_1")
        conclusions = [entry["conclusion"] for entry in result["reasoning_trace"]]
        self.assertIn("evaluation_analysis: pronunciation_quality", conclusions)
        self.assertIn("evaluation_analysis: mti_characteristics", conclusions)

    def test_scores_are_in_range_for_both_rubrics(self):
        result = synthesize_articulation(ISSUE_ARTICULATION_PACKET, "session_1")
        for rubric in ("pronunciation", "mti"):
            self.assertIn(result["scores"][rubric]["score"], (1, 2, 3, 4, 5))
            self.assertTrue(result["scores"][rubric]["reasoning"])

    def test_clean_evidence_scores_high(self):
        result = synthesize_articulation(CLEAN_ARTICULATION_PACKET, "session_1")
        self.assertEqual(result["scores"]["pronunciation"]["score"], 5)
        self.assertEqual(result["scores"]["mti"]["score"], 5)
        self.assertEqual(result["coach_output"]["overall_assessment"]["level"], "excellent")

    def test_severe_evidence_scores_low(self):
        result = synthesize_articulation(ISSUE_ARTICULATION_PACKET, "session_1")
        self.assertLessEqual(result["scores"]["pronunciation"]["score"], 2)

    def test_coach_output_session_id_and_coach_field(self):
        result = synthesize_articulation(ISSUE_ARTICULATION_PACKET, "session_42")
        self.assertEqual(result["coach_output"]["session_id"], "session_42")
        self.assertEqual(result["coach_output"]["coach"], "articulation")

    def test_findings_cite_real_evidence_words(self):
        result = synthesize_articulation(ISSUE_ARTICULATION_PACKET, "session_1")
        words_in_findings = {f["word"] for f in result["coach_output"]["detailed_findings"]}
        self.assertTrue(words_in_findings & {"will", "please"})

    def test_no_crash_on_empty_evidence_lists(self):
        result = synthesize_articulation(CLEAN_ARTICULATION_PACKET, "session_1")
        self.assertTrue(result["coach_output"]["strengths"])
        self.assertTrue(result["coach_output"]["practice_plan"])
        self.assertTrue(result["coach_output"]["next_session_focus"])


class SynthesizeDeliveryTests(unittest.TestCase):
    def test_output_has_the_five_top_level_keys_in_order(self):
        result = synthesize_delivery(CLEAN_DELIVERY_PACKET, "session_2")
        self.assertEqual(
            list(result.keys()),
            ["evaluation_analysis", "scores", "score_reasoning", "coach_output", "reasoning_trace"],
        )

    def test_evaluation_analysis_has_delivery_dimensions(self):
        result = synthesize_delivery(CLEAN_DELIVERY_PACKET, "session_2")
        self.assertEqual(
            set(result["evaluation_analysis"].keys()),
            {
                "speech_behavior_summary", "fluency_patterns", "rhythm", "intonation",
                "confidence_indicators", "engagement_patterns", "speaking_consistency",
            },
        )
        for text in result["evaluation_analysis"].values():
            self.assertTrue(text.strip())

    def test_evaluation_analysis_varies_with_evidence(self):
        clean = synthesize_delivery(CLEAN_DELIVERY_PACKET, "session_2")
        issue = synthesize_delivery(ISSUE_DELIVERY_PACKET, "session_2")
        self.assertNotEqual(clean["evaluation_analysis"]["fluency_patterns"], issue["evaluation_analysis"]["fluency_patterns"])
        self.assertNotEqual(clean["evaluation_analysis"]["engagement_patterns"], issue["evaluation_analysis"]["engagement_patterns"])

    def test_score_reasoning_covers_every_rubric_with_the_right_shape(self):
        result = synthesize_delivery(ISSUE_DELIVERY_PACKET, "session_2")
        self.assertEqual(set(result["score_reasoning"].keys()), {"fluency", "intonation", "engagement"})
        for rubric_reasoning in result["score_reasoning"].values():
            self.assertEqual(set(rubric_reasoning.keys()), _SCORE_REASONING_FIELDS)
            self.assertTrue(rubric_reasoning["justification"])
            self.assertTrue(rubric_reasoning["level1_evidence"])
            self.assertTrue(rubric_reasoning["level2_evidence"])
            self.assertTrue(rubric_reasoning["why_not_higher"])
            self.assertTrue(rubric_reasoning["why_not_lower"])

    def test_reasoning_trace_includes_evaluation_analysis_entries(self):
        result = synthesize_delivery(CLEAN_DELIVERY_PACKET, "session_2")
        conclusions = [entry["conclusion"] for entry in result["reasoning_trace"]]
        for dim in ("fluency_patterns", "rhythm", "intonation", "confidence_indicators", "engagement_patterns"):
            self.assertIn(f"evaluation_analysis: {dim}", conclusions)

    def test_scores_are_in_range_for_all_three_rubrics(self):
        result = synthesize_delivery(ISSUE_DELIVERY_PACKET, "session_2")
        for rubric in ("fluency", "intonation", "engagement"):
            self.assertIn(result["scores"][rubric]["score"], (1, 2, 3, 4, 5))
            self.assertTrue(result["scores"][rubric]["reasoning"])

    def test_clean_evidence_scores_high(self):
        result = synthesize_delivery(CLEAN_DELIVERY_PACKET, "session_2")
        self.assertGreaterEqual(result["scores"]["fluency"]["score"], 4)
        self.assertGreaterEqual(result["scores"]["intonation"]["score"], 4)
        self.assertGreaterEqual(result["scores"]["engagement"]["score"], 4)

    def test_severe_evidence_scores_low(self):
        result = synthesize_delivery(ISSUE_DELIVERY_PACKET, "session_2")
        self.assertLessEqual(result["scores"]["fluency"]["score"], 2)
        self.assertLessEqual(result["scores"]["intonation"]["score"], 2)
        self.assertLessEqual(result["scores"]["engagement"]["score"], 2)

    def test_coach_priority_fix_first_matches_top_priority_improvement(self):
        result = synthesize_delivery(ISSUE_DELIVERY_PACKET, "session_2")
        self.assertEqual(
            result["coach_output"]["coach_priority"]["fix_first"],
            result["coach_output"]["priority_improvements"][0]["issue"],
        )

    def test_coach_output_session_id_and_coach_field(self):
        result = synthesize_delivery(ISSUE_DELIVERY_PACKET, "session_99")
        self.assertEqual(result["coach_output"]["session_id"], "session_99")
        self.assertEqual(result["coach_output"]["coach"], "delivery")


class ClaudeTeacherProviderTests(unittest.TestCase):
    def test_generate_articulation_returns_valid_json_matching_the_contract(self):
        prompt = build_articulation_prompt(ISSUE_ARTICULATION_PACKET, "session_1")
        raw = ClaudeTeacherProvider().generate_articulation(prompt)
        parsed = json.loads(raw)
        self.assertEqual(set(parsed.keys()), {"evaluation_analysis", "scores", "score_reasoning", "coach_output", "reasoning_trace"})
        self.assertEqual(parsed["coach_output"]["session_id"], "session_1")

    def test_generate_delivery_returns_valid_json_matching_the_contract(self):
        prompt = build_delivery_prompt(ISSUE_DELIVERY_PACKET, "session_2")
        raw = ClaudeTeacherProvider().generate_delivery(prompt)
        parsed = json.loads(raw)
        self.assertEqual(set(parsed.keys()), {"evaluation_analysis", "scores", "score_reasoning", "coach_output", "reasoning_trace"})
        self.assertEqual(parsed["coach_output"]["session_id"], "session_2")

    def test_drop_in_compatible_with_teacher_runner(self):
        from backend.knowledge_distillation.teacher_generation.teacher_runner import TeacherRunner

        runner = TeacherRunner(provider=ClaudeTeacherProvider())
        raw = runner.run_articulation(CLEAN_ARTICULATION_PACKET, "session_1")
        self.assertEqual(json.loads(raw)["coach_output"]["coach"], "articulation")


class ModuleBoundaryTests(unittest.TestCase):
    def test_claude_teacher_never_calls_a_real_llm(self):
        """The whole point of this module is to avoid a network call --
        it must never import an LLM SDK/client, or it isn't actually the
        local, deterministic substitute it claims to be."""
        import ast

        forbidden = ("anthropic", "llm.client", "llm.anthropic_client", "openai")
        path = os.path.join(_PACKAGE_DIR, "claude_teacher.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        offending = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(phrase in name.lower() for phrase in forbidden):
                    offending.append(name)
        self.assertEqual(offending, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
