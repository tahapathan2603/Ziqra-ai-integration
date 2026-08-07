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

Usage (single GPU / CPU / MPS -- identical everywhere, one process):
    python -m backend.knowledge_distillation.training.trainer.train \\
        --config backend/knowledge_distillation/training/configs/articulation.yaml

    python -m backend.knowledge_distillation.training.trainer.train \\
        --config backend/knowledge_distillation/training/configs/delivery.yaml \\
        --auto-pull-ollama

Usage (multiple GPUs, e.g. Kaggle's T4x2 -- data-parallel via HF Accelerate,
launched as multiple processes; this file's own code is unchanged either
way, see the "Multi-GPU" note below):
    accelerate launch --multi_gpu --num_processes=2 \\
        -m backend.knowledge_distillation.training.trainer.train \\
        --config backend/knowledge_distillation/training/configs/articulation.yaml

## Multi-GPU (Accelerate / DistributedDataParallel, not DataParallel)

`transformers.Trainer` has native DDP support built in -- it activates
automatically when this script is *launched* as multiple processes (via
`accelerate launch` or `torchrun`), with no manual `DistributedDataParallel`
wrapping needed anywhere in this codebase. Each process gets a full model
replica and a different slice of the batch; gradients sync after each
backward pass. This is a THROUGHPUT optimization, not a memory one -- it
requires the full model to already fit on one GPU (see model_loader.py's
`resolve_effective_dtype` and this file's `gradient_checkpointing` wiring
for what makes that true on a 16GB T4). Model-parallel sharding
(`device_map="auto"`, which pools memory *across* GPUs) is a different,
largely incompatible-if-combined-naively technique, used only on the
quantized-loading path in model_loader.py -- not part of this DDP path.

Everything below that prints to the console or writes the final adapter is
guarded to run on the main process only (`_is_main_process()` /
`state.is_world_process_zero` in callbacks.py) -- under multiple processes,
unguarded prints/writes would duplicate N-ways or race on the same file.
Checkpoint saving during training is already rank-safe -- that's Trainer's
own, unmodified internal behavior, not something this codebase adds.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from transformers import Trainer, TrainingArguments

from . import PROJECT_ROOT, ConfigError, TrainingConfig, load_config
from .callbacks import RichTrainingCallback, build_callbacks
from .dataset import ConversationDataCollator, DatasetBuildError, build_datasets
from .model_loader import load_model_and_tokenizer, resolve_effective_dtype, select_device

logger = logging.getLogger(__name__)
console = Console()


def _is_main_process() -> bool:
    """True unless this is a non-zero-rank process under a multi-process
    launch (`accelerate launch`/`torchrun`, which set RANK on every
    process they spawn). Used for the preflight prints and final adapter
    save below, which happen before a Trainer/TrainingArguments object
    exists (state.is_world_process_zero, what callbacks.py uses, isn't
    available yet at that point) -- both checks agree once Trainer's own
    rank detection takes over, since both ultimately read the same
    launcher-set environment."""
    return int(os.environ.get("RANK", 0)) == 0


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
    # Same hardware-aware dtype decision model_loader.load_base_model uses --
    # sharing resolve_effective_dtype (rather than each re-deriving
    # bf16/fp16 from config.model.torch_dtype independently) is what
    # guarantees TrainingArguments' autocast precision can never disagree
    # with the dtype the weights actually loaded in.
    effective_dtype = resolve_effective_dtype(config.model.torch_dtype)
    device = select_device()

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
        bf16=(effective_dtype == torch.bfloat16),
        fp16=(effective_dtype == torch.float16),
        report_to=[],  # no wandb/tensorboard -- Rich (callbacks.py) is the terminal experience
        disable_tqdm=True,  # replaced by callbacks.py's Rich progress display
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,  # our Dataset returns exactly the keys the model expects
        dataloader_pin_memory=(device == "cuda"),  # pinned memory is CUDA-only; avoids a benign MPS/CPU warning
        # GPU memory: recompute activations during backward instead of
        # storing them -- see model_loader.attach_lora's comment for the
        # LoRA-specific prerequisite this pairs with
        # (enable_input_require_grads). use_reentrant=False is the modern,
        # non-reentrant checkpoint variant -- besides being the currently
        # recommended default, it's also what keeps this safe to combine
        # with DDP (the older reentrant variant has a known bad interaction
        # with DDP's gradient-ready hooks).
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.training.gradient_checkpointing else None,
        # GPU DataLoader throughput -- see TrainingHyperparams' docstring
        # in __init__.py for what each does and why the benefit is modest
        # here (tokenization is already eager, not on-the-fly).
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_persistent_workers=(
            config.training.dataloader_persistent_workers and config.training.dataloader_num_workers > 0
        ),  # persistent_workers requires num_workers > 0; guard rather than let Trainer raise
        dataloader_prefetch_factor=(
            config.training.dataloader_prefetch_factor if config.training.dataloader_num_workers > 0 else None
        ),  # same guard -- prefetch_factor is only meaningful with worker processes
        # Multi-GPU (DDP, see this module's docstring): every LoRA-targeted
        # layer participates in every forward pass (no conditional/early-exit
        # computation), and use_reentrant=False checkpointing avoids the
        # specific DDP+reentrant-checkpointing interaction that usually
        # motivates find_unused_parameters=True -- so False is set here for
        # DDP's extra per-step traversal to be skipped. If a "mark variable
        # ready only once" or "expected to have finished reduction" error
        # ever appears under multi-GPU, this is the first thing to flip.
        ddp_find_unused_parameters=False,
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
    is_main = _is_main_process()  # see this file's docstring -- guards duplicate N-way prints under multi-GPU

    if is_main:
        console.print(f"[bold]Loading model and tokenizer for '{config.dataset.coach}'...[/]")
    # Not rank-guarded: every process needs its own (model, tokenizer) --
    # this is what DDP means by "each GPU gets a full replica". HF Hub
    # downloads are themselves safe under concurrent multi-process access
    # (huggingface_hub locks the cache directory during download; a
    # process that finds the lock held waits rather than re-downloading).
    model, tokenizer = load_model_and_tokenizer(
        config.model, config.lora, config.quantization,
        auto_pull_ollama=args.auto_pull_ollama,
    )

    if is_main:
        console.print("[bold]Building datasets...[/]")
    try:
        train_dataset, val_dataset = build_datasets(
            config.dataset, tokenizer, config.training.max_sequence_length, project_root
        )
    except DatasetBuildError as e:
        console.print(f"[bold red]Dataset error:[/] {e}")
        raise SystemExit(1)

    output_dir = project_root / config.checkpointing.output_dir
    if is_main:
        _print_preflight_summary(config, project_root, len(train_dataset), len(val_dataset), output_dir)

    training_args = build_training_arguments(config, output_dir)
    data_collator = ConversationDataCollator(tokenizer)
    callbacks = build_callbacks(config, is_main_process=is_main)

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

    # Rank-guarded: under multi-GPU every process reaches this line, and an
    # unguarded save would have N processes writing the same files
    # concurrently -- harmless-looking but a real race (partial/torn
    # writes, not just duplicated work). trainer.is_world_process_zero()
    # is Trainer's own public API for exactly this check, used here
    # instead of _is_main_process()/RANK because by this point the
    # Trainer object exists and this is the more standard way to ask it.
    if trainer.is_world_process_zero():
        adapter_dir = output_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        console.print(f"[bold green]Final adapter saved to {adapter_dir}[/]")


if __name__ == "__main__":
    main()
