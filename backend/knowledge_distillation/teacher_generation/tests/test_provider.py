"""
Tests for the Teacher Communication Layer (provider.py). All use a fake
LLMClient double — no network call, no credentials required.
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

from backend.knowledge_distillation.teacher_generation import (
    ARTICULATION_ENV_PREFIX,
    DELIVERY_ENV_PREFIX,
    NonRetryableTeacherError,
    TeacherModelClient,
    TeacherProvider,
    TeacherProviderError,
)
from backend.llm.client import LLMRequestError
from backend.llm.config import LLMConfig


def make_llm_config(**overrides):
    defaults = dict(
        api_key="fake-key", base_url="https://example.invalid", model="fake-model",
        temperature=0.3, max_tokens=1024, timeout_seconds=60.0, max_retries=3,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


class FakeClient:
    """Stand-in for LLMClient -- only needs .complete() and .config."""

    def __init__(self, config, responses=None, error=None):
        self.config = config
        self._responses = list(responses) if responses is not None else None
        self._error = error
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._responses is not None:
            return self._responses.pop(0)
        return "a real response"


class TeacherModelClientTests(unittest.TestCase):
    def test_generate_returns_the_raw_response_text(self):
        client = FakeClient(make_llm_config(), responses=["hello from the teacher"])
        model_client = TeacherModelClient(role="test-model", env_prefix="ZIQRA_UNUSED_", client=client, base_retry_delay_seconds=0.0)
        self.assertEqual(model_client.generate("some prompt"), "hello from the teacher")
        self.assertEqual(client.calls, 1)

    def test_credits_error_raises_without_retrying(self):
        error = LLMRequestError(
            "LLM request failed (401): {'error': {'type': 'CreditsError', 'message': 'Insufficient balance.'}}"
        )
        client = FakeClient(make_llm_config(max_retries=3), error=error)
        model_client = TeacherModelClient(role="test-model", env_prefix="ZIQRA_UNUSED_", client=client, base_retry_delay_seconds=0.0)
        with self.assertRaises(NonRetryableTeacherError):
            model_client.generate("prompt")
        self.assertEqual(client.calls, 1)  # no retry loop for this error class

    def test_transient_error_retries_then_succeeds(self):
        class FlakyClient(FakeClient):
            def complete(self, prompt):
                self.calls += 1
                if self.calls < 2:
                    raise LLMRequestError("Connection error.")
                return "ok after retry"

        client = FlakyClient(make_llm_config(max_retries=3))
        model_client = TeacherModelClient(role="test-model", env_prefix="ZIQRA_UNUSED_", client=client, base_retry_delay_seconds=0.0)
        self.assertEqual(model_client.generate("prompt"), "ok after retry")
        self.assertEqual(client.calls, 2)

    def test_exhausted_retries_raises_teacher_provider_error(self):
        client = FakeClient(make_llm_config(max_retries=2), error=LLMRequestError("Connection error."))
        model_client = TeacherModelClient(role="test-model", env_prefix="ZIQRA_UNUSED_", client=client, base_retry_delay_seconds=0.0)
        with self.assertRaises(TeacherProviderError):
            model_client.generate("prompt")
        self.assertEqual(client.calls, 3)  # 1 + max_retries

    def test_empty_response_is_retried(self):
        client = FakeClient(make_llm_config(max_retries=2), responses=["", "", "finally, content"])
        model_client = TeacherModelClient(role="test-model", env_prefix="ZIQRA_UNUSED_", client=client, base_retry_delay_seconds=0.0)
        self.assertEqual(model_client.generate("prompt"), "finally, content")
        self.assertEqual(client.calls, 3)


class TeacherProviderTests(unittest.TestCase):
    def test_generate_articulation_uses_the_articulation_client(self):
        art_client = FakeClient(make_llm_config(), responses=["articulation output"])
        del_client = FakeClient(make_llm_config(), responses=["should not be called"])
        provider = TeacherProvider(
            articulation_client=TeacherModelClient(role="articulation", env_prefix=ARTICULATION_ENV_PREFIX, client=art_client),
            delivery_client=TeacherModelClient(role="delivery", env_prefix=DELIVERY_ENV_PREFIX, client=del_client),
        )
        result = provider.generate_articulation("prompt")
        self.assertEqual(result, "articulation output")
        self.assertEqual(art_client.calls, 1)
        self.assertEqual(del_client.calls, 0)

    def test_generate_delivery_uses_the_delivery_client(self):
        art_client = FakeClient(make_llm_config(), responses=["should not be called"])
        del_client = FakeClient(make_llm_config(), responses=["delivery output"])
        provider = TeacherProvider(
            articulation_client=TeacherModelClient(role="articulation", env_prefix=ARTICULATION_ENV_PREFIX, client=art_client),
            delivery_client=TeacherModelClient(role="delivery", env_prefix=DELIVERY_ENV_PREFIX, client=del_client),
        )
        result = provider.generate_delivery("prompt")
        self.assertEqual(result, "delivery output")
        self.assertEqual(del_client.calls, 1)
        self.assertEqual(art_client.calls, 0)

    def test_default_construction_does_not_require_credentials(self):
        """Building a TeacherProvider with no overrides must not touch the
        environment/network -- only an actual generate() call should."""
        provider = TeacherProvider()
        self.assertIsInstance(provider, TeacherProvider)


class ModuleBoundaryTests(unittest.TestCase):
    def test_provider_knows_nothing_about_coach_packets_or_prompts(self):
        """Communication only: provider.py must never import packet_builder,
        schemas (coach packet shapes), or anything prompt-construction-
        related. Locks in the Part 2 scope boundary."""
        import ast

        forbidden = ("packet_builder", "teacher_generation.schemas", "prompt_builder", "validator")
        path = os.path.join(_PACKAGE_DIR, "provider.py")
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
