"""
Centralizes which environment namespace identifies each teacher model's
credentials, model name, and generation tuning (temperature/max_tokens/
timeout/retries). Nothing is hardcoded in provider.py.

Reuses backend/llm/config.py's LLMConfig loader, and reuses the SAME env
prefixes the project's earlier LLM-evaluation-framework work already
established for these exact two models (see .env.example) — credentials
are entered once, not duplicated per consumer.
"""

from dataclasses import dataclass

from ...llm.config import LLMConfig

#: MiniMax M3 -> Articulation Coach
ARTICULATION_ENV_PREFIX = "ZIQRA_MINIMAX_"

#: MiMo-v2.5 -> Delivery Coach
DELIVERY_ENV_PREFIX = "ZIQRA_MIM_"


@dataclass(frozen=True)
class TeacherGenerationConfig:
    """Fully-resolved settings for both teacher models.

    Attributes:
        articulation: MiniMax M3's resolved config (credentials, model,
            temperature, max_tokens, timeout, retries) — see
            backend/llm/config.py's LLMConfig for the field list.
        delivery: MiMo-v2.5's resolved config, same shape.
    """

    articulation: LLMConfig
    delivery: LLMConfig

    @classmethod
    def from_env(cls) -> "TeacherGenerationConfig":
        """Resolve both models' configuration from the environment.

        Raises:
            LLMConfigError: if either model's required
                {prefix}API_KEY / {prefix}BASE_URL / {prefix}MODEL is unset.
        """
        return cls(
            articulation=LLMConfig.from_env(ARTICULATION_ENV_PREFIX),
            delivery=LLMConfig.from_env(DELIVERY_ENV_PREFIX),
        )
