"""
Reusable LLM client for OpenAI-compatible chat completion APIs.

Built on the official `openai` Python SDK rather than raw HTTP: the SDK
already implements the OpenAI-compatible request/response contract (messages,
retries, timeouts), so pointing `base_url` at any compatible provider works
unchanged — validated against OpenCode Go (base_url `https://opencode.ai/zen/v1`,
see .env.example), and equally usable for DeepSeek, OpenRouter, or a
self-hosted vLLM/Ollama server for the future student model
(backend/llm/student_model.py) — swapping providers is a .env change, never a
code change.

Public interface:
    LLMClient(config: Optional[LLMConfig] = None)
        .complete(user, system=None, **overrides) -> str
            Plain-text generation from a prompt, optionally with a system
            message. Use for any free-form generation need.
        .complete_json(user, system=None, validate=None, max_attempts=2, **overrides) -> dict
            Same as complete(), but parses the response as JSON and retries
            (with a corrective follow-up) if it isn't valid JSON or fails the
            caller's optional `validate` check. Use whenever a caller needs
            structured output rather than free text — e.g.
            backend/feedback/feedback_generator.py.

Every failure raises one of two exceptions, never an SDK-specific one:
    LLMConfigError  — setup problem (missing/invalid env var). Re-exported
                      from .config for convenience.
    LLMRequestError — the request failed (auth, rate limit, network, bad
                      response, or no valid JSON after retrying) — after the
                      SDK's own transport-level retries (max_retries) were
                      already exhausted.
"""

import json
import logging
import re
from typing import Callable, Dict, Optional

import openai

from .config import LLMConfig, LLMConfigError

logger = logging.getLogger(__name__)

__all__ = ["LLMClient", "LLMConfigError", "LLMRequestError", "extract_json"]

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMRequestError(Exception):
    """Raised when the LLM API call fails (after the SDK's own retries) or
    never yields valid JSON in complete_json()."""


def _strip_reasoning(content: str) -> str:
    """Reasoning models (e.g. DeepSeek R1) may emit <think>...</think> before
    the real answer. Strip it so callers never have to handle it."""
    return _THINK_TAG_RE.sub("", content).strip()


def extract_json(text: str) -> Optional[Dict]:
    """
    Best-effort JSON extraction from an LLM response: try a direct parse,
    then a fenced ```json block, then the outermost {...} substring.

    Shared by complete_json() and exposed directly since any caller doing its
    own ad-hoc JSON completion benefits from the same recovery logic rather
    than reimplementing it.
    """
    text = text.strip()
    for candidate in (text, *_FENCE_RE.findall(text)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


class LLMClient:
    """
    Reusable OpenAI-compatible chat client. See module docstring for the
    full interface.

    Config (API key, base URL, model, temperature, max_tokens, timeout,
    retries) is read from environment variables via LLMConfig.from_env() —
    nothing is hardcoded here. Pass an explicit `config` to override
    (e.g. from a test, or a future caller that needs a second provider).
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        # max_retries here covers transport-level failures (network blips,
        # 429/5xx) — the SDK retries those internally with backoff. That's a
        # distinct concern from complete_json()'s retry loop below, which
        # retries on semantically-invalid *content* (bad JSON), something the
        # SDK has no way to know about.
        self._client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )

    def _call(
        self,
        system: Optional[str],
        user: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        try:
            logger.info(f"Calling LLM ({self.config.model})...")
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            )
        except openai.APIStatusError as e:
            # The response body usually carries the provider's actual error
            # message (e.g. "Insufficient balance", "Invalid API key") in
            # e.body — surface it explicitly rather than relying on str(e)
            # alone, so failures are actionable without a separate curl probe.
            detail = e.body if getattr(e, "body", None) else str(e)
            raise LLMRequestError(f"LLM request failed ({e.status_code}): {detail}") from e
        except openai.APIError as e:
            raise LLMRequestError(f"LLM request failed: {e}") from e

        content = response.choices[0].message.content or ""
        return _strip_reasoning(content)

    def complete(
        self,
        user: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a plain-text response. `system` is optional — omit it for a
        bare prompt completion, or pass it to steer behavior/persona.
        `temperature`/`max_tokens` override the configured defaults for this
        call only.

        Raises LLMRequestError if the call fails after the SDK's own retries.
        """
        return self._call(system, user, temperature, max_tokens)

    def complete_json(
        self,
        user: str,
        system: Optional[str] = None,
        validate: Optional[Callable[[Dict], bool]] = None,
        max_attempts: int = 2,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Generate a structured JSON response. If the response isn't valid JSON,
        or fails the optional `validate` callback (e.g. checking required
        schema keys), retries with a corrective follow-up message up to
        `max_attempts` times total.

        Raises LLMRequestError if the underlying call fails, or if no valid
        JSON response is obtained within max_attempts.
        """
        raw = self._call(system, user, temperature, max_tokens)
        parsed = extract_json(raw)

        attempt = 1
        while (parsed is None or (validate and not validate(parsed))) and attempt < max_attempts:
            logger.warning(f"LLM response was not valid JSON (attempt {attempt}) — retrying with a correction.")
            corrective = (
                user
                + "\n\nYour previous response was not valid:\n"
                + raw
                + "\n\nReturn ONLY a single valid JSON object matching the required schema. "
                "No prose, no markdown fences."
            )
            raw = self._call(system, corrective, temperature, max_tokens)
            parsed = extract_json(raw)
            attempt += 1

        if parsed is None or (validate and not validate(parsed)):
            raise LLMRequestError(f"LLM did not return valid JSON after {attempt} attempt(s). Last response: {raw}")

        return parsed
