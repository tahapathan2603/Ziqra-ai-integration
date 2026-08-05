"""
Tests for teacher_generation.prompt_builder. All local/pure -- string
assembly only, no network call, no coach-packet/provider dependency.
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

from backend.knowledge_distillation.teacher_generation.prompt_builder import (
    ARTICULATION_RUBRICS,
    ARTICULATION_TEACHER,
    DELIVERY_RUBRICS,
    DELIVERY_TEACHER,
    build_articulation_prompt,
    build_delivery_prompt,
)

ARTICULATION_PACKET = {
    "coach": "articulation",
    "pronunciation": {
        "score": None,
        "phoneme_accuracy_pct": 90,
        "observations": ["Overall pronunciation is strong."],
        "mispronounced_words": [{"word": "will", "severity": "high", "start": 17.66, "end": 17.84}],
    },
    "mti": {
        "score": None,
        "clarity_impact": "medium",
        "top_recurring_issue": "consonant substitution",
        "patterns_detected": ["consonant substitution"],
        "clarity_risk_words": ["will"],
        "observations": [],
        "patterns": [],
    },
}

DELIVERY_PACKET = {
    "coach": "delivery",
    "fluency": {
        "speaking_speed": {"words_per_minute": 226, "classification": "too_fast"},
        "filler_usage": {"count": 4, "per_minute": 13.1, "classification": "needs_improvement", "examples": []},
        "pauses": {"total": 5, "dead_air_count": 0, "long_pause_count": 0, "hesitation_count": 1},
    },
    "intonation": {
        "score": None, "delivery_label": "Expressive", "observations": [],
        "pitch": {"average_hz": 129.1, "range_hz": 65.5}, "flat_sections": [],
    },
    "rhythm": {"score_pct": 7, "issues": []},
    "engagement": {"score": None, "level": "needs_improvement"},
}


class ArticulationPromptTests(unittest.TestCase):
    def test_returns_a_string(self):
        prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_1")
        self.assertIsInstance(prompt, str)

    def test_names_the_correct_teacher(self):
        prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_1")
        self.assertIn(ARTICULATION_TEACHER, prompt)
        self.assertNotIn(DELIVERY_TEACHER, prompt)

    def test_names_only_its_own_rubrics(self):
        prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_1")
        for rubric in ARTICULATION_RUBRICS:
            self.assertIn(f'"{rubric}"', prompt)
        for rubric in DELIVERY_RUBRICS:
            self.assertNotIn(f'"{rubric}"', prompt)

    def test_output_contract_keys_present_in_order(self):
        prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_1")
        self.assertLess(prompt.index('"scores"'), prompt.index('"coach_output"'))
        self.assertLess(prompt.index('"coach_output"'), prompt.index('"reasoning_trace"'))

    def test_coach_output_schema_keys_present(self):
        prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_1")
        for key in ("overall_assessment", "priority_improvements", "detailed_findings",
                    "recurring_patterns", "practice_plan", "review_timeline", "next_session_focus"):
            self.assertIn(key, prompt)

    def test_evidence_packet_embedded_verbatim(self):
        prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_42")
        self.assertIn("session_42", prompt)
        self.assertIn(json.dumps(ARTICULATION_PACKET, indent=2, ensure_ascii=False), prompt)

    def test_no_hallucination_instruction_present(self):
        prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_1")
        self.assertIn("Do not invent", prompt)

    def test_duplication_is_intentional_note_present(self):
        prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_1")
        self.assertIn("duplication is intentional", prompt)


class DeliveryPromptTests(unittest.TestCase):
    def test_returns_a_string(self):
        prompt = build_delivery_prompt(DELIVERY_PACKET, "session_2")
        self.assertIsInstance(prompt, str)

    def test_names_the_correct_teacher(self):
        prompt = build_delivery_prompt(DELIVERY_PACKET, "session_2")
        self.assertIn(DELIVERY_TEACHER, prompt)
        self.assertNotIn(ARTICULATION_TEACHER, prompt)

    def test_names_only_its_own_rubrics(self):
        prompt = build_delivery_prompt(DELIVERY_PACKET, "session_2")
        for rubric in DELIVERY_RUBRICS:
            self.assertIn(f'"{rubric}"', prompt)
        for rubric in ARTICULATION_RUBRICS:
            self.assertNotIn(f'"{rubric}"', prompt)

    def test_output_contract_keys_present_in_order(self):
        prompt = build_delivery_prompt(DELIVERY_PACKET, "session_2")
        self.assertLess(prompt.index('"scores"'), prompt.index('"coach_output"'))
        self.assertLess(prompt.index('"coach_output"'), prompt.index('"reasoning_trace"'))

    def test_coach_output_schema_keys_present(self):
        prompt = build_delivery_prompt(DELIVERY_PACKET, "session_2")
        for key in ("overall_assessment", "interviewer_impression", "priority_improvements",
                    "detailed_findings", "timeline_review", "behavioral_patterns",
                    "practice_plan", "coach_priority", "next_session_focus"):
            self.assertIn(key, prompt)

    def test_evidence_packet_embedded_verbatim(self):
        prompt = build_delivery_prompt(DELIVERY_PACKET, "session_99")
        self.assertIn("session_99", prompt)
        self.assertIn(json.dumps(DELIVERY_PACKET, indent=2, ensure_ascii=False), prompt)


class ModuleBoundaryTests(unittest.TestCase):
    def test_prompt_builder_is_independent_of_provider_and_packet_builder(self):
        """prompt_builder.py's only job is Coach Packet -> Prompt. It must
        not import provider.py (the teacher runner's transport) or
        packet_builder.py (coach packet derivation) -- both would couple
        prompt construction to modules it's explicitly meant to stay
        independent of."""
        import ast

        forbidden = ("teacher_generation.provider", "teacher_generation.packet_builder")
        path = os.path.join(_PACKAGE_DIR, "prompt_builder.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        offending = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # relative imports like "from .provider import X" have
                # module="provider", level=1 -- normalize to the dotted form
                # the forbidden list checks against.
                module = node.module or ""
                names = [f"teacher_generation.{module}" if node.level else module]
            else:
                continue
            for name in names:
                if any(phrase in name for phrase in forbidden):
                    offending.append(name)
        self.assertEqual(offending, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
