"""
Tests for teacher_generation.teacher_runner. Uses a fake TeacherProvider
double -- no network call, no credentials required.
"""

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
    build_articulation_prompt,
    build_delivery_prompt,
)
from backend.knowledge_distillation.teacher_generation.teacher_runner import TeacherRunner

ARTICULATION_PACKET = {
    "coach": "articulation",
    "pronunciation": {"score": None, "phoneme_accuracy_pct": 90, "observations": [], "mispronounced_words": []},
    "mti": {"score": None, "clarity_impact": "low", "top_recurring_issue": None, "patterns_detected": [], "clarity_risk_words": [], "observations": [], "patterns": []},
}

DELIVERY_PACKET = {
    "coach": "delivery",
    "fluency": {"speaking_speed": {"words_per_minute": 130, "classification": "ideal"}, "filler_usage": {"count": 0, "per_minute": 0.0, "classification": "good", "examples": []}, "pauses": {"total": 0, "dead_air_count": 0, "long_pause_count": 0, "hesitation_count": 0}},
    "intonation": {"score": None, "delivery_label": "Expressive", "observations": [], "pitch": {"average_hz": 140.0, "range_hz": 50.0}, "flat_sections": []},
    "rhythm": {"score_pct": 90, "issues": []},
    "engagement": {"score": None, "level": "good"},
}


class FakeProvider:
    """Stand-in for TeacherProvider -- records every prompt it receives."""

    def __init__(self, articulation_response="articulation raw response", delivery_response="delivery raw response"):
        self._articulation_response = articulation_response
        self._delivery_response = delivery_response
        self.articulation_prompts = []
        self.delivery_prompts = []

    def generate_articulation(self, prompt: str) -> str:
        self.articulation_prompts.append(prompt)
        return self._articulation_response

    def generate_delivery(self, prompt: str) -> str:
        self.delivery_prompts.append(prompt)
        return self._delivery_response


class RunArticulationTests(unittest.TestCase):
    def test_returns_the_providers_raw_response(self):
        provider = FakeProvider(articulation_response="raw text from MiniMax M3")
        runner = TeacherRunner(provider=provider)
        result = runner.run_articulation(ARTICULATION_PACKET, "session_1")
        self.assertEqual(result, "raw text from MiniMax M3")

    def test_sends_the_prompt_builder_output_to_the_provider(self):
        provider = FakeProvider()
        runner = TeacherRunner(provider=provider)
        runner.run_articulation(ARTICULATION_PACKET, "session_1")
        expected_prompt = build_articulation_prompt(ARTICULATION_PACKET, "session_1")
        self.assertEqual(provider.articulation_prompts, [expected_prompt])

    def test_never_calls_the_delivery_client(self):
        provider = FakeProvider()
        runner = TeacherRunner(provider=provider)
        runner.run_articulation(ARTICULATION_PACKET, "session_1")
        self.assertEqual(provider.delivery_prompts, [])


class RunDeliveryTests(unittest.TestCase):
    def test_returns_the_providers_raw_response(self):
        provider = FakeProvider(delivery_response="raw text from MiMo-v2.5")
        runner = TeacherRunner(provider=provider)
        result = runner.run_delivery(DELIVERY_PACKET, "session_2")
        self.assertEqual(result, "raw text from MiMo-v2.5")

    def test_sends_the_prompt_builder_output_to_the_provider(self):
        provider = FakeProvider()
        runner = TeacherRunner(provider=provider)
        runner.run_delivery(DELIVERY_PACKET, "session_2")
        expected_prompt = build_delivery_prompt(DELIVERY_PACKET, "session_2")
        self.assertEqual(provider.delivery_prompts, [expected_prompt])

    def test_never_calls_the_articulation_client(self):
        provider = FakeProvider()
        runner = TeacherRunner(provider=provider)
        runner.run_delivery(DELIVERY_PACKET, "session_2")
        self.assertEqual(provider.articulation_prompts, [])


class DefaultConstructionTests(unittest.TestCase):
    def test_default_construction_does_not_require_credentials(self):
        """Building a TeacherRunner with no overrides must not touch the
        environment/network -- only an actual run_* call should (and even
        then, only via the provider it delegates to)."""
        runner = TeacherRunner()
        self.assertIsInstance(runner, TeacherRunner)


class ModuleBoundaryTests(unittest.TestCase):
    def test_runner_does_not_duplicate_packet_building_or_api_clients(self):
        """teacher_runner.py orchestrates prompt_builder + provider only.
        It must never import packet_builder (building coach packets is not
        its job), an LLM SDK/client directly (that would be a duplicate API
        client -- provider.py is the sole transport), or json (no parsing
        happens here)."""
        import ast

        forbidden = ("packet_builder", "llm.client", "llm.anthropic_client", "json")
        path = os.path.join(_PACKAGE_DIR, "teacher_runner.py")
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
                if any(phrase in name for phrase in forbidden):
                    offending.append(name)
        self.assertEqual(offending, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
