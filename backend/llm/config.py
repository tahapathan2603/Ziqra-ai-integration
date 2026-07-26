"""
Environment-driven configuration for the LLM integration layer.

All values come from environment variables — nothing here is hardcoded, so
switching providers (OpenCode Go, DeepSeek, OpenRouter, a local
OpenAI-compatible server, or later the fine-tuned student model in
backend/llm/student_model.py) is purely a matter of setting different env
vars, never editing code.

Loads a project-root .env file (if present) via python-dotenv before reading
os.environ, so a local `.env` works the same way `export`-ing the variables
in a shell would. See .env.example at the project root for the full variable
list and instructions on where to get an OpenCode Go API key.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Same explicit-project-root-path pattern used by preprocessing/silero_vad.py,
# rather than relying on python-dotenv's cwd-dependent search — this way
# config loading works the same regardless of which directory a script is
# run from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))

_ENV_PREFIX = "ZIQRA_LLM_"

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2


class LLMConfigError(Exception):
    """Raised when required LLM configuration is missing or invalid."""


def _require(prefix: str, name: str) -> str:
    value = os.environ.get(prefix + name)
    if not value:
        raise LLMConfigError(
            f"{prefix}{name} is not set. Copy .env.example to .env at the project root "
            f"and fill in {prefix}{name} (see .env.example for where to find each value)."
        )
    return value


def _optional_float(prefix: str, name: str, default: float) -> float:
    value = os.environ.get(prefix + name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        raise LLMConfigError(f"{prefix}{name} must be a number, got: {value!r}")


def _optional_int(prefix: str, name: str, default: int) -> int:
    value = os.environ.get(prefix + name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise LLMConfigError(f"{prefix}{name} must be an integer, got: {value!r}")


@dataclass(frozen=True)
class LLMConfig:
    """
    Fully-resolved LLM provider configuration. Construct via from_env() —
    the dataclass itself carries no defaults for api_key/base_url/model
    since those must always come from the environment, never from code.
    """

    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_env(cls, prefix: str = _ENV_PREFIX) -> "LLMConfig":
        """
        Build config from environment variables (see .env.example). With the
        default prefix (used by the live feedback path, backend/feedback/
        feedback_generator.py):
            ZIQRA_LLM_API_KEY          (required)
            ZIQRA_LLM_BASE_URL         (required)
            ZIQRA_LLM_MODEL            (required)
            ZIQRA_LLM_TEMPERATURE      (optional, default 0.3)
            ZIQRA_LLM_MAX_TOKENS       (optional, default 1024)
            ZIQRA_LLM_TIMEOUT_SECONDS  (optional, default 60)
            ZIQRA_LLM_MAX_RETRIES      (optional, default 2)

        A different `prefix` (e.g. "ZIQRA_TEACHER_") reads the same variable
        names under that prefix instead — this is how the teacher-model client
        (backend/llm/anthropic_client.py) reuses this exact class/loader for a
        second, independently configured provider, without duplicating this file.

        Raises LLMConfigError if a required variable is missing, or an
        optional numeric one is set but not parseable.
        """
        return cls(
            api_key=_require(prefix, "API_KEY"),
            base_url=_require(prefix, "BASE_URL"),
            model=_require(prefix, "MODEL"),
            temperature=_optional_float(prefix, "TEMPERATURE", DEFAULT_TEMPERATURE),
            max_tokens=_optional_int(prefix, "MAX_TOKENS", DEFAULT_MAX_TOKENS),
            timeout_seconds=_optional_float(prefix, "TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_retries=_optional_int(prefix, "MAX_RETRIES", DEFAULT_MAX_RETRIES),
        )
