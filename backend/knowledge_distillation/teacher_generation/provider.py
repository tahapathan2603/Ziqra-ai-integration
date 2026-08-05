"""
Teacher Communication Layer. Abstracts all API interaction with the two
teacher models so the rest of the codebase never calls an LLM SDK directly:

    MiniMax M3   -> Articulation Coach
    MiMo-v2.5     -> Delivery Coach

Communication only. This file does NOT know about Level 1, Level 2, coach
packets, prompt construction, JSON schemas, or validation — it accepts a
prompt and returns raw text, nothing else. Building the prompt is a later
module's job; parsing/validating the response is a later module's job.

Concrete transport reuses the project's existing OpenAI-compatible client
(backend/llm/client.py) — both models sit behind an OpenAI-compatible
chat.completions endpoint (see backend/distillation/providers/
minimax_provider.py and mim_provider.py, the earlier, proven integration
for these same two models). Credentials/model/tuning are resolved via
config.py, never hardcoded here.
"""

import logging
import time
from typing import Optional

from ...llm.client import LLMClient, LLMRequestError
from ...llm.config import LLMConfig, LLMConfigError
from .config import ARTICULATION_ENV_PREFIX, DELIVERY_ENV_PREFIX
from .exceptions import NonRetryableTeacherError, TeacherProviderError

logger = logging.getLogger(__name__)

# First backoff interval; each subsequent retry doubles it.
BASE_RETRY_DELAY_SECONDS = 1.0

# Substrings (case-insensitive) that mark an account-level block rather than
# a transient failure — skip the backoff loop entirely and raise
# immediately. See NonRetryableTeacherError's docstring for why this exists.
NON_RETRYABLE_ERROR_MARKERS = (
    "creditserror",
    "insufficient balance",
    "invalid api key",
    "invalid x-api-key",
    "authentication_error",
)


class TeacherModelClient:
    """Communicates with ONE teacher model.

    Generic over which model — parameterized by an env prefix (or an
    explicit LLMConfig), not hardcoded to MiniMax or MiMo. TeacherProvider
    below composes two of these; nothing about this class is specific to
    either.
    """

    def __init__(
        self,
        role: str,
        env_prefix: str,
        config: Optional[LLMConfig] = None,
        client: Optional[LLMClient] = None,
        base_retry_delay_seconds: float = BASE_RETRY_DELAY_SECONDS,
    ) -> None:
        """
        Args:
            role: Human label for logging (e.g. "articulation (MiniMax M3)").
            env_prefix: Which ZIQRA_* namespace to resolve credentials/model/
                tuning from, if `config` isn't given directly.
            config: Optional pre-resolved LLMConfig, e.g. for tests or a
                caller that already has one. Resolved lazily from
                `env_prefix` if omitted — building a client never requires
                credentials to be present.
            client: Optional pre-built LLMClient, injected by tests.
            base_retry_delay_seconds: First backoff interval (doubles each
                retry). Overridable so tests can set it to 0 and run at full
                speed instead of actually sleeping through backoff.
        """
        self._role = role
        self._env_prefix = env_prefix
        self._config = config
        self._client = client
        self._base_retry_delay_seconds = base_retry_delay_seconds

    def _get_client(self) -> LLMClient:
        if self._client is None:
            try:
                self._client = LLMClient(self._config or LLMConfig.from_env(self._env_prefix))
            except LLMConfigError as e:
                raise NonRetryableTeacherError(f"{self._role} is not configured: {e}") from e
        return self._client

    def generate(self, prompt: str) -> str:
        """Send `prompt` to this model, retrying transient failures with
        exponential backoff, and return the raw response text.

        Raises:
            NonRetryableTeacherError: the failure is account-level (no
                credits, bad credentials) — raised immediately, no retries.
            TeacherProviderError: once every attempt is exhausted.
        """
        client = self._get_client()
        total_attempts = client.config.max_retries + 1
        delay = self._base_retry_delay_seconds
        last_error: Optional[str] = None

        for attempt in range(1, total_attempts + 1):
            try:
                logger.info("Calling %s (%s, attempt %d/%d)...", self._role, client.config.model, attempt, total_attempts)
                response = client.complete(prompt)
            except LLMRequestError as e:
                message = str(e)
                if any(marker in message.lower() for marker in NON_RETRYABLE_ERROR_MARKERS):
                    logger.error("Non-retryable failure from %s, aborting immediately: %s", self._role, message)
                    raise NonRetryableTeacherError(message) from e
                last_error = message
                logger.warning("%s call failed (attempt %d/%d): %s", self._role, attempt, total_attempts, e)
            else:
                if response.strip():
                    return response
                last_error = f"{self._role} returned an empty response"
                logger.warning("%s returned an empty response (attempt %d/%d).", self._role, attempt, total_attempts)

            if attempt < total_attempts:
                logger.info("Retrying in %.1fs...", delay)
                time.sleep(delay)
                delay *= 2

        raise TeacherProviderError(
            f"{self._role} produced no usable response after {total_attempts} attempt(s). Last error: {last_error}"
        )


class TeacherProvider:
    """Clean entry point for the rest of the codebase: two named methods,
    one per coach. Nothing downstream needs to know these are two different
    models behind two different clients.
    """

    def __init__(
        self,
        articulation_client: Optional[TeacherModelClient] = None,
        delivery_client: Optional[TeacherModelClient] = None,
    ) -> None:
        """
        Args:
            articulation_client: Override for the MiniMax M3 client, e.g.
                to inject a fake in tests. Defaults to a real
                TeacherModelClient resolved from ARTICULATION_ENV_PREFIX.
            delivery_client: Same, for MiMo-v2.5 / DELIVERY_ENV_PREFIX.
        """
        self._articulation = articulation_client or TeacherModelClient(
            role="articulation (MiniMax M3)", env_prefix=ARTICULATION_ENV_PREFIX
        )
        self._delivery = delivery_client or TeacherModelClient(
            role="delivery (MiMo-v2.5)", env_prefix=DELIVERY_ENV_PREFIX
        )

    def generate_articulation(self, prompt: str) -> str:
        """Send `prompt` to MiniMax M3 and return its raw response text.

        Raises:
            NonRetryableTeacherError / TeacherProviderError: see
                TeacherModelClient.generate.
        """
        return self._articulation.generate(prompt)

    def generate_delivery(self, prompt: str) -> str:
        """Send `prompt` to MiMo-v2.5 and return its raw response text.

        Raises:
            NonRetryableTeacherError / TeacherProviderError: see
                TeacherModelClient.generate.
        """
        return self._delivery.generate(prompt)
