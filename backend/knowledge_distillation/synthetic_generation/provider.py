"""
The generation model, fully abstracted behind `Provider.generate()`.

No concrete implementation lives here. For this phase, Claude (via this
Claude Code session) is the synthetic data generator — content is authored
in-session and ingested through pipeline.ingest_authored(), which never
calls this interface at all. Qwen must not be introduced into this
pipeline: it is a student model, trained later, after the distillation
dataset already exists — not a generator for it. A concrete Provider (e.g.
a genuine Claude API client, once real credentials exist) can be added
later and plugged into EvidenceGenerator/Pipeline without changing either;
nothing in this module should default-wire to any external LLM in the
meantime.

Nothing outside a future concrete Provider should touch an LLM SDK.
Authentication, request construction, retries, timeouts, and logging
belong there; `generate()` returns the response body as raw text. It
deliberately does NOT build prompts, parse JSON, or validate anything —
those belong to prompt_builder.py, evidence_generator.py, and
validator.py respectively.
"""

from abc import ABC, abstractmethod

from .exceptions import NonRetryableProviderError, ProviderError  # noqa: F401 -- re-exported for implementers

# Substrings (case-insensitive) a future concrete Provider should treat as
# an account-level block rather than a transient failure -- raise
# NonRetryableProviderError immediately on a match, no backoff loop. Learned
# the hard way on a different provider: retrying "CreditsError: Insufficient
# balance" on nearly every call ran for 7 hours before being stopped
# manually. No amount of retrying an exhausted balance or revoked key ever
# succeeds.
NON_RETRYABLE_ERROR_MARKERS = (
    "creditserror",
    "insufficient balance",
    "invalid api key",
    "invalid x-api-key",
    "authentication_error",
)


class Provider(ABC):
    """Interface every generation backend implements.

    One method by design — a caller (and a test double) needs nothing more
    than prompt in, raw text out.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send `prompt` to the model and return its raw response text.

        Raises:
            NonRetryableProviderError: an account-level block.
            ProviderError: every retry was exhausted.
        """
        raise NotImplementedError
