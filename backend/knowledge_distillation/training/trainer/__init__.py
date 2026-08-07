"""
Supervised fine-tuning trainer for the two knowledge-distillation student
models (Part 10): Articulation -> Gemma, Delivery -> Qwen.

    training/configs/{coach}.yaml
            |
            v
    load_config()                     (this module)
            |
            v
    TrainingConfig
            |
      ,-----+------+---------+---------------+------------.
      v     v      v         v               v            v
   model  dataset  lora   training      checkpointing  randomness
      |
      v
    model_loader.py    -- Ollama presence-check/pull, HF model+tokenizer, LoRA
    dataset.py          -- training.data + training.prompts -> tokenized datasets
    metrics.py           -- validation-time metrics
    callbacks.py          -- Rich terminal UI, early stopping
    train.py               -- orchestrates all of the above; the only entry point

This __init__.py holds the one thing every submodule needs -- the typed
config schema matching training/configs/*.yaml, and its loader -- the same
pattern training.data (DistillationSample) and training.prompts (template
loading) already use: shared logic lives in __init__.py, each script
imports from here rather than redefining it.

Nothing in this package trains anything by importing it; `train.py`'s
`if __name__ == "__main__"` block is the only thing that starts a run.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# backend/knowledge_distillation/training/trainer/__init__.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class ConfigError(Exception):
    """`training/configs/*.yaml` is missing, isn't valid YAML, or is
    missing a required section/field."""


@dataclass(frozen=True)
class ModelConfig:
    """`model:` section -- see training/configs/README.md."""

    model_name: str
    tokenizer_name: Optional[str]
    trust_remote_code: bool
    torch_dtype: str
    ollama_model_name: Optional[str] = None


@dataclass(frozen=True)
class QuantizationConfig:
    """`quantization:` section. CUDA + bitsandbytes only; ignored (with a
    warning) on any other device -- see model_loader.py."""

    enabled: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    """`dataset:` section. `train_file`/`validation_file`/`test_file` are
    repo-root-relative paths to the (both-coach-combined) split JSONL
    files training.data.split_dataset produces; `coach` selects this
    config's subset of them."""

    coach: str
    train_file: str
    validation_file: str
    test_file: str


@dataclass(frozen=True)
class LoRAConfig:
    """`lora:` section."""

    enabled: bool
    rank: int
    alpha: int
    dropout: float
    target_modules: Tuple[str, ...]


@dataclass(frozen=True)
class TrainingHyperparams:
    """`training:` section."""

    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    max_sequence_length: int
    eval_accumulation_steps: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    # GPU memory optimization: trades ~20-30% more compute time for a large
    # cut in activation memory (recomputes activations during backward
    # instead of storing them) -- see model_loader.attach_lora's comment
    # for why this requires enable_input_require_grads() on a LoRA model.
    gradient_checkpointing: bool = True
    # GPU DataLoader throughput: workers > 0 lets batch collation happen on
    # CPU while the GPU is still busy with the previous step, instead of
    # the two waiting on each other serially. persistent_workers avoids
    # respawning worker processes every epoch; prefetch_factor controls how
    # many batches each worker stages ahead of time. See dataset.py's
    # ConversationDataset docstring for why the benefit here is smaller
    # than for on-the-fly (e.g. image/audio) datasets -- tokenization is
    # already done eagerly, so there's little CPU work left to overlap.
    dataloader_num_workers: int = 2
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2


@dataclass(frozen=True)
class CheckpointConfig:
    """`checkpointing:` section. `output_dir` is the run's root; the
    trainer creates checkpoints/, adapters/, and logs/ underneath it."""

    output_dir: str
    save_strategy: str
    save_steps: int
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int


@dataclass(frozen=True)
class RandomnessConfig:
    """`randomness:` section."""

    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    """One fully-parsed `training/configs/{coach}.yaml`."""

    model: ModelConfig
    dataset: DatasetConfig
    lora: LoRAConfig
    training: TrainingHyperparams
    checkpointing: CheckpointConfig
    randomness: RandomnessConfig
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


def _require_section(raw: dict, name: str, path: Path) -> dict:
    if name not in raw:
        raise ConfigError(f"{path}: missing top-level section '{name}'")
    return raw[name]


def load_config(path: Path) -> TrainingConfig:
    """Parse one `training/configs/*.yaml` file into a typed TrainingConfig.

    Raises:
        ConfigError: the file is missing, isn't valid YAML, or is missing
            a required section/field.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML ({e})") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a top-level YAML mapping, got {type(raw).__name__}")

    model_raw = _require_section(raw, "model", path)
    dataset_raw = _require_section(raw, "dataset", path)
    lora_raw = _require_section(raw, "lora", path)
    training_raw = _require_section(raw, "training", path)
    checkpoint_raw = _require_section(raw, "checkpointing", path)
    randomness_raw = _require_section(raw, "randomness", path)
    quant_raw = raw.get("quantization", {})  # optional -- defaults to disabled

    try:
        return TrainingConfig(
            model=ModelConfig(
                model_name=model_raw["model_name"],
                tokenizer_name=model_raw.get("tokenizer_name"),
                trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
                torch_dtype=model_raw.get("torch_dtype", "bfloat16"),
                ollama_model_name=model_raw.get("ollama_model_name"),
            ),
            dataset=DatasetConfig(
                coach=dataset_raw["coach"],
                train_file=dataset_raw["train_file"],
                validation_file=dataset_raw["validation_file"],
                test_file=dataset_raw["test_file"],
            ),
            lora=LoRAConfig(
                enabled=bool(lora_raw.get("enabled", True)),
                rank=int(lora_raw["rank"]),
                alpha=int(lora_raw["alpha"]),
                dropout=float(lora_raw["dropout"]),
                target_modules=tuple(lora_raw["target_modules"]),
            ),
            training=TrainingHyperparams(
                epochs=int(training_raw["epochs"]),
                learning_rate=float(training_raw["learning_rate"]),
                batch_size=int(training_raw["batch_size"]),
                gradient_accumulation_steps=int(training_raw["gradient_accumulation_steps"]),
                warmup_ratio=float(training_raw["warmup_ratio"]),
                weight_decay=float(training_raw["weight_decay"]),
                max_sequence_length=int(training_raw["max_sequence_length"]),
                eval_accumulation_steps=training_raw.get("eval_accumulation_steps"),
                early_stopping_patience=training_raw.get("early_stopping_patience"),
                gradient_checkpointing=bool(training_raw.get("gradient_checkpointing", True)),
                dataloader_num_workers=int(training_raw.get("dataloader_num_workers", 2)),
                dataloader_persistent_workers=bool(training_raw.get("dataloader_persistent_workers", True)),
                dataloader_prefetch_factor=int(training_raw.get("dataloader_prefetch_factor", 2)),
            ),
            checkpointing=CheckpointConfig(
                output_dir=checkpoint_raw["output_dir"],
                save_strategy=checkpoint_raw["save_strategy"],
                save_steps=int(checkpoint_raw["save_steps"]),
                logging_steps=int(checkpoint_raw["logging_steps"]),
                evaluation_strategy=checkpoint_raw["evaluation_strategy"],
                eval_steps=int(checkpoint_raw["eval_steps"]),
            ),
            randomness=RandomnessConfig(seed=int(randomness_raw["seed"])),
            quantization=QuantizationConfig(
                enabled=bool(quant_raw.get("enabled", False)),
                load_in_4bit=bool(quant_raw.get("load_in_4bit", True)),
                bnb_4bit_compute_dtype=quant_raw.get("bnb_4bit_compute_dtype", "bfloat16"),
                bnb_4bit_quant_type=quant_raw.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(quant_raw.get("bnb_4bit_use_double_quant", True)),
            ),
        )
    except KeyError as e:
        raise ConfigError(f"{path}: missing required field {e}") from e


__all__ = [
    "PROJECT_ROOT",
    "CheckpointConfig",
    "ConfigError",
    "DatasetConfig",
    "LoRAConfig",
    "ModelConfig",
    "QuantizationConfig",
    "RandomnessConfig",
    "TrainingConfig",
    "TrainingHyperparams",
    "load_config",
]
