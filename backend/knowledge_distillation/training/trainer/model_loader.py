"""
Model loading for student training (Part 10).

Everything related to "how do we end up with a ready-to-train (model,
tokenizer) pair" lives here -- Ollama presence-checking, Hugging Face
model/tokenizer loading, quantization, and LoRA attachment. Every other
module in this package (dataset.py, callbacks.py, train.py) works with a
plain (model, tokenizer) pair and never needs to know where the weights
came from. Swapping the base model, the quantization scheme, or dropping
Ollama entirely should only ever require changes inside this one file.

## Why Ollama only checks/pulls, and never loads training weights

Ollama stores models as quantized GGUF blobs for its own llama.cpp-based
inference runtime. That format is not what `transformers`/PEFT load for
gradient-based fine-tuning: different tensor layout, a lossy/one-way
quantization, and no autograd graph. So `OllamaManager` here does exactly
one job -- confirm the model is pulled locally (or pull it), purely as a
convenience/consistency check -- and the trainable weights always come
from `AutoModelForCausalLM.from_pretrained(model_cfg.model_name)` (the
Hugging Face Hub or a local path). This is also why the config carries
`model.model_name` (HF id) and `model.ollama_model_name` (Ollama tag) as
two separate fields -- the two ecosystems name the same model differently.
A missing/unreachable Ollama installation never blocks training; it only
downgrades to a warning, since HF loading is independent of it.
"""

import json
import logging
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from . import LoRAConfig as LoRAConfigSpec
from . import ModelConfig as ModelConfigSpec
from . import QuantizationConfig as QuantizationConfigSpec

logger = logging.getLogger(__name__)

_OLLAMA_HOST = "http://localhost:11434"
_OLLAMA_TIMEOUT_SECONDS = 2.0

_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Ollama: presence-check + optional pull only (see module docstring)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OllamaStatus:
    """Result of checking one model against the local Ollama install."""

    daemon_reachable: bool
    model_present: bool
    checked_name: Optional[str]


class OllamaManager:
    """Thin wrapper around the local Ollama daemon's REST API (presence
    check) and CLI (pull). Never raises on an unreachable/missing Ollama
    installation -- see module docstring for why that's safe."""

    def __init__(self, host: str = _OLLAMA_HOST, timeout: float = _OLLAMA_TIMEOUT_SECONDS) -> None:
        self._host = host
        self._timeout = timeout

    def is_daemon_reachable(self) -> bool:
        try:
            urllib.request.urlopen(f"{self._host}/api/tags", timeout=self._timeout)
            return True
        except (urllib.error.URLError, OSError):
            return False

    def has_model(self, ollama_model_name: str) -> bool:
        """True if `ollama_model_name` (e.g. "gemma3:4b") is already
        pulled locally. False (with a logged warning) if the daemon
        can't be reached at all."""
        try:
            with urllib.request.urlopen(f"{self._host}/api/tags", timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            logger.warning("Could not reach Ollama daemon to check for '%s': %s", ollama_model_name, e)
            return False
        names = {m.get("name") for m in data.get("models", [])}
        return ollama_model_name in names

    def pull(self, ollama_model_name: str) -> bool:
        """Shell out to `ollama pull <name>`, streaming Ollama's own
        progress output straight to the terminal. Returns True on success,
        False (logged, never raised) if the CLI isn't on PATH or the pull
        fails."""
        logger.info("Pulling '%s' via Ollama -- this can take a while for a multi-GB model.", ollama_model_name)
        try:
            result = subprocess.run(["ollama", "pull", ollama_model_name], check=False)
        except FileNotFoundError:
            logger.error("`ollama` CLI not found on PATH -- cannot pull '%s'.", ollama_model_name)
            return False
        return result.returncode == 0

    def ensure_model(self, ollama_model_name: Optional[str], auto_pull: bool) -> OllamaStatus:
        """Check `ollama_model_name` against the local install (a no-op if
        it's None) and either pull it (if missing and `auto_pull`) or print
        instructions for the user to pull it themselves.

        Never raises: an unreachable daemon, a missing CLI, or a failed
        pull all degrade to a warning, since training proceeds via
        Hugging Face regardless of Ollama's local state.
        """
        if ollama_model_name is None:
            return OllamaStatus(daemon_reachable=False, model_present=False, checked_name=None)

        if not self.is_daemon_reachable():
            logger.warning(
                "Ollama daemon not reachable at %s -- skipping the '%s' presence check. "
                "Training proceeds via Hugging Face regardless.",
                self._host, ollama_model_name,
            )
            return OllamaStatus(daemon_reachable=False, model_present=False, checked_name=ollama_model_name)

        if self.has_model(ollama_model_name):
            logger.info("Ollama model '%s' is present locally.", ollama_model_name)
            return OllamaStatus(daemon_reachable=True, model_present=True, checked_name=ollama_model_name)

        if auto_pull:
            logger.info("Ollama model '%s' not found locally; auto-pulling (--auto-pull-ollama)...", ollama_model_name)
            pulled = self.pull(ollama_model_name)
            return OllamaStatus(daemon_reachable=True, model_present=pulled, checked_name=ollama_model_name)

        logger.warning(
            "Ollama model '%s' is not pulled locally. Run `ollama pull %s` yourself, or re-run "
            "with --auto-pull-ollama. Continuing -- training loads weights via Hugging Face "
            "either way.",
            ollama_model_name, ollama_model_name,
        )
        return OllamaStatus(daemon_reachable=True, model_present=False, checked_name=ollama_model_name)


# ---------------------------------------------------------------------------
# Hugging Face model/tokenizer loading + LoRA
# ---------------------------------------------------------------------------
def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_BY_NAME:
        raise ValueError(f"Unknown torch_dtype '{name}'. Supported: {sorted(_DTYPE_BY_NAME)}")
    return _DTYPE_BY_NAME[name]


def select_device() -> str:
    """Best available device, in preference order: CUDA, Apple MPS, CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_quantization_config(quant_cfg: QuantizationConfigSpec):
    """A `transformers.BitsAndBytesConfig`, or None if quantization is
    disabled or unsupported on this machine (no CUDA, or bitsandbytes
    isn't installed) -- logged, never a hard failure, since quantization
    is a memory optimization, not a correctness requirement."""
    if not quant_cfg.enabled:
        return None
    if not torch.cuda.is_available():
        logger.warning("quantization.enabled is true but no CUDA GPU is available; loading unquantized.")
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning("quantization.enabled is true but bitsandbytes is not installed; loading unquantized.")
        return None

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=_resolve_dtype(quant_cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )


def load_tokenizer(model_cfg: ModelConfigSpec) -> PreTrainedTokenizerBase:
    """Load the tokenizer named by `model_cfg.tokenizer_name`, falling
    back to `model_cfg.model_name`. Adds a pad token (aliased to EOS) if
    the base tokenizer doesn't define one -- common for Llama-family
    tokenizers, and required for batched training."""
    tokenizer_name = model_cfg.tokenizer_name or model_cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=model_cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_cfg: ModelConfigSpec, quant_cfg: QuantizationConfigSpec) -> PreTrainedModel:
    """Load the base causal-LM from the Hugging Face Hub (or a local
    path), at `model_cfg.torch_dtype`, optionally quantized, moved to the
    best available device."""
    dtype = _resolve_dtype(model_cfg.torch_dtype)
    quantization_config = _build_quantization_config(quant_cfg)

    load_kwargs = dict(trust_remote_code=model_cfg.trust_remote_code, dtype=dtype)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = "auto"  # required by bitsandbytes-quantized loading

    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **load_kwargs)

    if quantization_config is None:
        model = model.to(select_device())
    return model


def attach_lora(model: PreTrainedModel, lora_cfg: LoRAConfigSpec) -> PreTrainedModel:
    """Attach trainable LoRA adapters. `peft.get_peft_model` freezes every
    base parameter automatically (`requires_grad=False`) and leaves only
    the newly-added adapter matrices trainable -- nothing here needs to
    freeze anything manually. If `lora_cfg.enabled` is False, every base
    parameter is left trainable instead (full fine-tuning fallback)."""
    if not lora_cfg.enabled:
        logger.info("lora.enabled is false; training full model weights (no adapters attached).")
        for param in model.parameters():
            param.requires_grad = True
        return model

    peft_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(
    model_cfg: ModelConfigSpec,
    lora_cfg: LoRAConfigSpec,
    quant_cfg: QuantizationConfigSpec,
    ollama_manager: Optional[OllamaManager] = None,
    auto_pull_ollama: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """The single entry point the rest of the trainer calls: Ollama
    presence-check -> tokenizer -> base model -> LoRA. Returns a
    ready-to-train (model, tokenizer) pair.
    """
    manager = ollama_manager or OllamaManager()
    manager.ensure_model(model_cfg.ollama_model_name, auto_pull=auto_pull_ollama)

    tokenizer = load_tokenizer(model_cfg)
    model = load_base_model(model_cfg, quant_cfg)
    model = attach_lora(model, lora_cfg)
    return model, tokenizer


__all__ = [
    "OllamaManager",
    "OllamaStatus",
    "attach_lora",
    "load_base_model",
    "load_model_and_tokenizer",
    "load_tokenizer",
    "select_device",
]
