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
from typing import Dict, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

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


def resolve_effective_dtype(requested_name: str) -> torch.dtype:
    """GPU-aware mixed-precision selection: bf16 if the active CUDA device's
    tensor cores actually support it, fp16 otherwise -- "BF16 if supported,
    otherwise FP16", decided from real hardware capability
    (`torch.cuda.is_bf16_supported()`), not assumed from the config alone.

    Concretely: NVIDIA T4 (Turing, compute capability 7.5 -- Kaggle's and
    Colab's free-tier GPU) has no bf16 tensor cores; only Ampere and newer
    (A100, RTX 30xx+, compute capability 8.0+) do. Requesting bf16 on a T4
    doesn't error -- it silently runs unaccelerated, quietly losing the
    speed half of "mixed precision" while keeping only the memory-layout
    half. This function is called from both load_base_model (what dtype
    the weights load in) and train.py's build_training_arguments (what
    dtype TrainingArguments.bf16/fp16 tell Trainer's autocast to use) so
    the two can never disagree about which precision is actually active.

    Only overrides on CUDA -- Apple MPS and CPU have different (and, for
    this project's Mac dev machine, already-verified-working) bf16
    characteristics that this T4-specific check has no evidence about.
    """
    dtype = _resolve_dtype(requested_name)
    if dtype is torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning(
            "torch_dtype 'bfloat16' requested, but this CUDA device has no bf16 tensor-core "
            "support (e.g. T4/Turing) -- falling back to float16 so mixed precision is actually "
            "accelerated, not just memory-shaped."
        )
        return torch.float16
    return dtype


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
        bnb_4bit_compute_dtype=resolve_effective_dtype(quant_cfg.bnb_4bit_compute_dtype),  # same T4-vs-Ampere check as the base model dtype
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


def resolve_text_only_config(model_cfg: ModelConfigSpec):
    """The text-tower config for a multimodal checkpoint, or None if the
    checkpoint is already text-only.

    This project trains text-only students, but some checkpoints are
    multimodal: `google/gemma-3-4b-it` carries a SigLIP vision tower, and
    plain `AutoModelForCausalLM.from_pretrained` on it builds
    `Gemma3ForConditionalGeneration` -- costing memory twice over on a 16GB
    T4, confirmed by a real OOM:

      1. 453M vision-tower parameters are loaded and never used (the model
         is 4.333B params as a multimodal model vs 3.880B text-only).
      2. Much worse, `Gemma3ForConditionalGeneration.forward` runs its own
         loss rather than the shared `ForCausalLMLoss`, and that path makes
         a THIRD full copy of the logits:
             logits.float()                              # fp32 copy
             shift_logits[shift_attention_mask != 0].contiguous()  # another
         With Gemma-3's 262,208-token vocabulary that extra copy is
         seq * 262208 * 4 bytes -- 3.75 GB at a ~3570-token example, which
         is exactly the allocation that failed. `Gemma3ForCausalLM` calls
         `self.loss_function(...)` instead and never makes it. Note the
         gather is pure waste here regardless: at batch_size 1 there is no
         padding, so the mask is all ones and it copies the whole tensor to
         select all of it.

    Passing this config to `from_pretrained` makes the Auto class resolve to
    the text-only architecture (`Gemma3TextConfig` -> `Gemma3ForCausalLM`),
    which loads the checkpoint's `language_model.*` tensors cleanly -- its
    `base_model_prefix` is already "language_model", and every parameter
    name matches with nothing missing or left over (verified against the
    real checkpoint index).

    Detection is generic rather than a Gemma special case: `get_text_config()`
    returns a *different* object for a composite/multimodal config and
    `self` for an already-text-only one, so Qwen2.5 (delivery) takes the
    unchanged path.
    """
    config = AutoConfig.from_pretrained(
        model_cfg.model_name, trust_remote_code=model_cfg.trust_remote_code
    )
    text_config = config.get_text_config()
    if text_config is config:
        return None
    return text_config


def _gemma_text_only_key_mapping() -> Optional[Dict[str, str]]:
    """Whether loading a multimodal-saved Gemma-3 checkpoint into the
    text-only `Gemma3ForCausalLM` class needs an explicit `key_mapping` to
    land the real language-model weights (vs. silently reinitializing
    them) -- and what that mapping is, if so.

    `google/gemma-3-4b-it`'s checkpoint stores every language-model tensor
    under a `language_model.model.*` / `language_model.lm_head.*` prefix
    (saved from the multimodal `Gemma3ForConditionalGeneration` class --
    this is the exact legacy naming transformers' own
    `Gemma3ForConditionalGeneration._checkpoint_conversion_mapping`
    documents and corrects for that class). `Gemma3ForCausalLM`'s own real
    attributes are flat -- `self.model` (`Gemma3TextModel`) and a
    top-level `self.lm_head` -- so those checkpoint keys need translating
    down to `model.*` / `lm_head.*` to match.

    Whether `from_pretrained` does that translation AUTOMATICALLY (via its
    own `base_model_prefix`-driven key-renaming, no `key_mapping` needed)
    or requires one to be supplied explicitly is NOT stable across
    transformers versions -- confirmed by actually running both against a
    synthetic checkpoint reproducing the real prefix naming, not by
    reading changelogs:

      - transformers==4.57.6 (this project's own requirements.txt pin):
        `Gemma3ForCausalLM.base_model_prefix == "language_model"`. The
        automatic renaming already produces the correct `model.*` keys
        with NO `key_mapping` -- passing one here actively BREAKS it: the
        automatic step still runs afterward, tries to strip
        `language_model.` a second time from keys that no longer have it,
        and silently drops every one as missing/unexpected instead.
      - transformers==5.0.0 (what Kaggle's `train_on_kaggle.ipynb` Cell 1
        actually installs -- `!pip install -q transformers ...` is
        unpinned, and this is what it resolved to as of 2026-08): the same
        class's `base_model_prefix` is now `"model"` (an upstream fix
        matching the real attribute name -- but one that flips which
        automatic branch fires). The automatic path no longer produces the
        right keys at all: every `language_model.*` key is reported
        unexpected and every real `model.*`/`lm_head.*` key reported
        missing, silently reinitializing the entire 3.88B-parameter
        language model -- this is the exact "UNEXPECTED:
        language_model.model.*" / "MISSING: model.*" / "This checkpoint
        seem corrupted" failure observed on a real Kaggle T4x2 run. An
        explicit `key_mapping` IS required here, and this exact one has
        been verified (0 missing, 0 unexpected, every tensor bit-exact,
        `device_map="auto"` included) against a synthetic checkpoint on a
        real transformers==5.0.0 install.

    Branching on the actually-installed `Gemma3ForCausalLM`'s own
    `base_model_prefix` (not on `transformers.__version__`) means this
    keeps working if some future release changes it again, rather than
    hardcoding a version cutoff this project never pinned in the first
    place -- Kaggle's install is deliberately unpinned (see that
    notebook's Cell 1) and is not something this file controls.
    """
    from transformers import Gemma3ForCausalLM

    if Gemma3ForCausalLM.base_model_prefix == "language_model":
        return None  # verified: automatic renaming already handles this correctly (4.57.6)
    return {"^language_model.model": "model", "^language_model.lm_head": "lm_head"}  # verified against 5.0.0


def _assert_language_model_weights_loaded(loading_info: dict, model_name: str) -> None:
    """Raise if a text-only-narrowed load left any real language-model
    weight missing (i.e. reinitialized instead of loaded from the
    checkpoint) -- see `_gemma_text_only_key_mapping`'s docstring for the
    exact failure mode this catches. `unexpected_keys` (the skipped
    vision_tower/multi_modal_projector tensors) are expected and fine --
    only `missing_keys` under the text-only model's own `model.`/`lm_head.`
    prefixes indicate weights that should have loaded but didn't.

    A quiet reinitialization here is far worse than a loud failure: training
    would proceed on a partially-random ~3.88B-parameter base model --
    LoRA has nothing real to adapt, and results would be silently garbage
    rather than an obvious crash.
    """
    missing = set(loading_info.get("missing_keys") or [])
    substantial_missing = {k for k in missing if k.startswith("model.") or k.startswith("lm_head.")}
    if substantial_missing:
        raise RuntimeError(
            f"Loading {model_name} as a text-only checkpoint left "
            f"{len(substantial_missing)} language-model weight(s) missing "
            f"(reinitialized instead of loaded from the checkpoint), e.g. "
            f"{sorted(substantial_missing)[:5]}. This means "
            f"_gemma_text_only_key_mapping's key_mapping no longer matches "
            f"this checkpoint/transformers version -- do not proceed with "
            f"training on a partially-random base model. See that "
            f"function's docstring for the versions already verified."
        )


def load_base_model(model_cfg: ModelConfigSpec, quant_cfg: QuantizationConfigSpec) -> PreTrainedModel:
    """Load the base causal-LM from the Hugging Face Hub (or a local
    path), at `model_cfg.torch_dtype` (hardware-adjusted -- see
    resolve_effective_dtype), optionally quantized, moved to the best
    available device. Multimodal checkpoints are narrowed to their text
    tower -- see resolve_text_only_config."""
    dtype = resolve_effective_dtype(model_cfg.torch_dtype)
    quantization_config = _build_quantization_config(quant_cfg)

    load_kwargs = dict(trust_remote_code=model_cfg.trust_remote_code, dtype=dtype)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = "auto"  # required by bitsandbytes-quantized loading

    text_config = resolve_text_only_config(model_cfg)
    if text_config is not None:
        logger.info(
            "%s is a multimodal checkpoint; loading its text tower only "
            "(drops the unused vision encoder and avoids the multimodal loss path's "
            "extra full-size logits copy).",
            model_cfg.model_name,
        )
        load_kwargs["config"] = text_config
        key_mapping = _gemma_text_only_key_mapping()
        if key_mapping is not None:
            load_kwargs["key_mapping"] = key_mapping

        # See _assert_language_model_weights_loaded's docstring: verify
        # the real weights actually landed rather than trusting
        # from_pretrained's version-dependent defaults silently.
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name, output_loading_info=True, **load_kwargs
        )
        _assert_language_model_weights_loaded(loading_info, model_cfg.model_name)
    else:
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

    # GPU memory optimization: required for gradient checkpointing to work
    # correctly on a LoRA model, and harmless if checkpointing ends up
    # disabled. With the base model frozen (requires_grad=False on every
    # non-adapter parameter), gradient checkpointing's recompute-on-backward
    # has no live tensor to hook a gradient onto at the checkpoint boundary
    # unless the INPUT to the checkpointed segment is explicitly marked as
    # requiring grad -- this call does exactly that (on the embedding
    # output). Without it, checkpointing + LoRA can silently fail to
    # backprop into the adapters at all (no error -- loss.backward() runs,
    # LoRA gradients just stay None), rather than raise. Actual
    # gradient_checkpointing_enable() is called by transformers.Trainer
    # itself, driven by TrainingArguments.gradient_checkpointing (see
    # train.py's build_training_arguments) -- not called here, since this
    # module doesn't own the Trainer/TrainingArguments lifecycle.
    model.enable_input_require_grads()
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

    # QLoRA prerequisite. Gated on the model's OWN post-load flag rather
    # than `quant_cfg.enabled`, because _build_quantization_config silently
    # declines to quantize on a machine without CUDA or without bitsandbytes
    # (both are logged warnings, not errors) -- so the config flag can be
    # true while the loaded model is plain fp16/bf16, and calling this on an
    # unquantized model would be wrong.
    #
    # What it does: casts layernorms and the embedding/output layers back to
    # fp32 and makes the frozen 4-bit base safe to backprop through. Without
    # it, QLoRA training is prone to silent instability (NaN/diverging loss)
    # rather than a clean failure. It also calls enable_input_require_grads()
    # -- attach_lora does the same immediately after, which is idempotent.
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        logger.info("Quantized base detected; applying prepare_model_for_kbit_training (QLoRA).")
        # use_gradient_checkpointing=False: Trainer owns that decision and
        # calls gradient_checkpointing_enable() itself, driven by
        # TrainingArguments.gradient_checkpointing (see train.py). Letting
        # this helper also enable it would apply the default *reentrant*
        # variant, silently overriding the use_reentrant=False that train.py
        # deliberately sets for DDP compatibility.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    model = attach_lora(model, lora_cfg)
    return model, tokenizer


__all__ = [
    "OllamaManager",
    "OllamaStatus",
    "attach_lora",
    "load_base_model",
    "load_model_and_tokenizer",
    "load_tokenizer",
    "resolve_effective_dtype",
    "resolve_text_only_config",
    "select_device",
]
