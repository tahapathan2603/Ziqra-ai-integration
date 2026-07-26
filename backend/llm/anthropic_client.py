"""
Reusable LLM client for Anthropic Messages-API-compatible providers.

Built on the official `anthropic` Python SDK — validated against OpenCode Go's
"Go" endpoint (base_url `https://opencode.ai/zen/go/v1`, model `qwen3.7-plus`;
see .env.example). This is a separate client from client.py (which speaks the
OpenAI-compatible chat.completions protocol, still used unchanged by the live
feedback path, backend/feedback/feedback_generator.py) because the two APIs
have genuinely different request/response shapes:

  - Anthropic: `system` is a top-level request parameter, never a message with
    role "system"; the response's `content` is a LIST of typed blocks (e.g.
    {"type": "text", "text": ...}, {"type": "thinking", "thinking": ...}) rather
    than a single `choices[0].message.content` string.
  - Extended-thinking models here return their reasoning as its own
    {"type": "thinking"} block, cleanly separate from the {"type": "text"}
    answer block — unlike DeepSeek R1's inline "<think>...</think>" tags
    (see client.py's _strip_reasoning), so no regex stripping is needed: this
    client simply ignores non-"text" blocks and joins the "text" ones.

This client is intended for the teacher/parent model role in the distillation
pipeline (see backend/distillation/, the audio-model-architecture plan) — NOT
the live single-LLM feedback path, which keeps using client.py/qwen3.6-plus
unchanged. Config is read via the shared LLMConfig loader (backend/llm/config.py)
using the "ZIQRA_TEACHER_" prefix, so both clients' credentials coexist in one
.env without collision.

Public interface mirrors client.py exactly (LLMClient), so callers can treat
the two as interchangeable:
    AnthropicLLMClient(config: Optional[LLMConfig] = None)
        .complete(user, system=None, **overrides) -> str
        .complete_json(user, system=None, validate=None, max_attempts=2, **overrides) -> dict

Every failure raises one of two exceptions, never an SDK-specific one:
    LLMConfigError  — setup problem (missing/invalid env var).
    LLMRequestError — the request failed, or no valid JSON after retrying.
"""

import logging
from typing import Callable, Dict, Optional

import anthropic

from .client import LLMRequestError, extract_json
from .config import LLMConfig, LLMConfigError

logger = logging.getLogger(__name__)

__all__ = ["AnthropicLLMClient", "LLMConfigError", "LLMRequestError"]

TEACHER_ENV_PREFIX = "ZIQRA_TEACHER_"


class AnthropicLLMClient:
    """
    Reusable Anthropic-Messages-compatible chat client. See module docstring
    for the full interface and why this is separate from client.py's LLMClient.

    Config defaults to the "ZIQRA_TEACHER_" env prefix (distinct from the live
    feedback path's "ZIQRA_LLM_" prefix) via LLMConfig.from_env(). Pass an
    explicit `config` to override (e.g. from a test).
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env(prefix=TEACHER_ENV_PREFIX)
        self._client = anthropic.Anthropic(
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
        try:
            logger.info(f"Calling teacher LLM ({self.config.model})...")
            response = self._client.messages.create(
                model=self.config.model,
                system=system if system else anthropic.NOT_GIVEN,
                # content as an explicit block list, not a plain string: some
                # providers routed through this gateway (confirmed: Xiaomi's
                # MiMo) reject a bare string content with a generic "messages
                # must not be empty" error even though the request is well-
                # formed — the block-list form is Anthropic's canonical shape
                # (a plain string is only ever a shorthand for it), so this is
                # strictly more compatible, not a per-provider special case.
                messages=[{"role": "user", "content": [{"type": "text", "text": user}]}],
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            )
        except anthropic.APIStatusError as e:
            # The response body usually carries the provider's actual error
            # message in e.body — surface it explicitly, same discipline as
            # client.py's handling of openai.APIStatusError.
            detail = e.body if getattr(e, "body", None) else str(e)
            raise LLMRequestError(f"Teacher LLM request failed ({e.status_code}): {detail}") from e
        except anthropic.APIError as e:
            raise LLMRequestError(f"Teacher LLM request failed: {e}") from e

        # content is a list of typed blocks (text / thinking / ...); keep only
        # the text ones — thinking arrives as its own block here, not inline
        # tags, so there's nothing to strip out of the text itself.
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "".join(text_blocks).strip()

    def complete(
        self,
        user: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a plain-text response. `system` is optional. `temperature`/
        `max_tokens` override the configured defaults for this call only.

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
        or fails the optional `validate` callback, retries with a corrective
        follow-up message up to `max_attempts` times total. Identical retry
        logic to client.py's LLMClient.complete_json — see that docstring.
        """
        raw = self._call(system, user, temperature, max_tokens)
        parsed = extract_json(raw)

        attempt = 1
        while (parsed is None or (validate and not validate(parsed))) and attempt < max_attempts:
            logger.warning(f"Teacher LLM response was not valid JSON (attempt {attempt}) — retrying with a correction.")
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
            raise LLMRequestError(f"Teacher LLM did not return valid JSON after {attempt} attempt(s). Last response: {raw}")

        return parsed
