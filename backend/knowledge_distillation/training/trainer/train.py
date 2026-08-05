"""
CLI orchestrator for supervised fine-tuning of one student model
(Articulation -> Gemma, Delivery -> Qwen), driven by one
`training/configs/*.yaml` file.

Pure orchestration: every actual responsibility -- model loading
(including the Ollama presence-check), dataset construction, the terminal
UI, metrics -- is delegated to this package's other modules. This file
only parses arguments, loads the config, calls each module in order, and
prints the pre-flight summary before training starts.

Platform-independent by construction, not by special-casing: nothing here
(or in model_loader.py/dataset.py/callbacks.py) imports or checks for any
specific execution environment. `PROJECT_ROOT` is derived from this
file's own location on disk (`Path(__file__).resolve().parents[4]`), so
every config-relative path (dataset splits, checkpoint output_dir) resolves
correctly no matter where the repo happens to be cloned or mounted --
`/Users/you/Ziqra.ai` locally, `/content/Ziqra.ai` on Colab, a RunPod/
Lambda volume, or a Kaggle working directory. Device selection
(model_loader.select_device()) probes for CUDA/MPS/CPU at runtime instead
of assuming one. Running this script is the same one command everywhere:
    python training/trainer/train.py --config training/configs/<coach>.yaml
A notebook or remote host that wants to use this trainer is responsible
for getting the repo onto disk and a Python environment with the right
packages installed -- not for anything training-related, which stays here.

Usage:
    python -m backend.knowledge_distillation.training.trainer.train \\
        --config backend/knowledge_distillation/training/configs/articulation.yaml

    python -m backend.knowledge_distillation.training.trainer.train \\
        --config backend/knowledge_distillation/training/configs/delivery.yaml \\
        --auto-pull-ollama
"""

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table
from transformers import Trainer, TrainingArguments

from . import PROJECT_ROOT, ConfigError, TrainingConfig, load_config
from .callbacks import RichTrainingCallback, build_callbacks
from .dataset import ConversationDataCollator, DatasetBuildError, build_datasets
from .model_loader import load_model_and_tokenizer, select_device

logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="path to a training/configs/*.yaml file")
    parser.add_argument(
        "--auto-pull-ollama", action="store_true",
        help="automatically `ollama pull` the configured model if it isn't present locally "
             "(default: print instructions and continue -- see model_loader.py)",
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="repo root that config-relative paths resolve against (default: auto-detected)",
    )
    return parser.parse_args()


def _print_preflight_summary(
    config: TrainingConfig, project_root: Path, train_size: int, val_size: int, output_dir: Path
) -> None:
    """Everything the CLI section asks to see before training starts:
    loaded configuration, selected model, dataset sizes, output directory."""
    table = Table(title="Training Configuration", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Coach", config.dataset.coach)
    table.add_row("Model", config.model.model_name)
    table.add_row("Tokenizer", config.model.tokenizer_name or "(same as model)")
    table.add_row("LoRA", f"enabled (r={config.lora.rank}, alpha={config.lora.alpha})" if config.lora.enabled else "disabled")
    table.add_row("Quantization", "enabled" if config.quantization.enabled else "disabled")
    table.add_row("Train samples", str(train_size))
    table.add_row("Validation samples", str(val_size))
    table.add_row("Epochs", str(config.training.epochs))
    table.add_row("Effective batch size", str(config.training.batch_size * config.training.gradient_accumulation_steps))
    table.add_row("Max sequence length", str(config.training.max_sequence_length))
    table.add_row("Output directory", str(output_dir))
    console.print(table)


def build_training_arguments(config: TrainingConfig, output_dir: Path) -> TrainingArguments:
    """`training/configs/*.yaml`'s `training`/`checkpointing`/`randomness`
    sections, translated into `transformers.TrainingArguments`. Kept as
    its own function (rather than inline in `main`) so the mapping from
    config field to library argument is visible in one place -- useful
    since library argument names occasionally drift across versions
    (e.g. `evaluation_strategy` -> `eval_strategy`) while this config's
    field names stay stable.
    """
    return TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        logging_dir=str(output_dir / "logs"),
        num_train_epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        eval_accumulation_steps=config.training.eval_accumulation_steps,
        save_strategy=config.checkpointing.save_strategy,
        save_steps=config.checkpointing.save_steps,
        logging_steps=config.checkpointing.logging_steps,
        eval_strategy=config.checkpointing.evaluation_strategy,
        eval_steps=config.checkpointing.eval_steps,
        seed=config.randomness.seed,
        bf16=(config.model.torch_dtype == "bfloat16"),
        fp16=(config.model.torch_dtype == "float16"),
        report_to=[],  # no wandb/tensorboard -- Rich (callbacks.py) is the terminal experience
        disable_tqdm=True,  # replaced by callbacks.py's Rich progress display
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,  # our Dataset returns exactly the keys the model expects
        dataloader_pin_memory=(select_device() == "cuda"),  # pinned memory is CUDA-only; avoids a benign MPS/CPU warning
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        config = load_config(args.config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        raise SystemExit(1)
    project_root = args.project_root or PROJECT_ROOT

    console.print(f"[bold]Loading model and tokenizer for '{config.dataset.coach}'...[/]")
    model, tokenizer = load_model_and_tokenizer(
        config.model, config.lora, config.quantization,
        auto_pull_ollama=args.auto_pull_ollama,
    )

    console.print("[bold]Building datasets...[/]")
    try:
        train_dataset, val_dataset = build_datasets(
            config.dataset, tokenizer, config.training.max_sequence_length, project_root
        )
    except DatasetBuildError as e:
        console.print(f"[bold red]Dataset error:[/] {e}")
        raise SystemExit(1)

    output_dir = project_root / config.checkpointing.output_dir
    _print_preflight_summary(config, project_root, len(train_dataset), len(val_dataset), output_dir)

    training_args = build_training_arguments(config, output_dir)
    data_collator = ConversationDataCollator(tokenizer)
    callbacks = build_callbacks(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Trainer's on_train_end callback hook (which normally stops the Rich
    # Live display) only fires on successful completion -- not reliably on
    # a crash inside a training step. Without this, a mid-training
    # exception leaves the Live display's background refresh thread
    # running, redrawing the full status panel on top of the traceback as
    # it prints. Stopping it explicitly here guarantees a clean traceback
    # regardless of how training ends.
    rich_callback = next((cb for cb in callbacks if isinstance(cb, RichTrainingCallback)), None)
    try:
        trainer.train()
    finally:
        if rich_callback is not None:
            rich_callback.stop()

    adapter_dir = output_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    console.print(f"[bold green]Final adapter saved to {adapter_dir}[/]")


if __name__ == "__main__":
    main()
