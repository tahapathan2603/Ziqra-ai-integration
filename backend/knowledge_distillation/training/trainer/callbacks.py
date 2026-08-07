"""
Training callbacks for student training (Part 10) -- the terminal UI, and
early stopping.

`RichTrainingCallback` is a `transformers.TrainerCallback`: Trainer calls
its `on_*` hooks at the matching point in the training loop, and it
renders a live-updating Rich display in response -- a progress bar plus a
status panel (epoch/step, losses, perplexity, learning rate, elapsed/ETA,
samples processed, current + best checkpoint, GPU/memory). Nothing here
runs the training loop itself; `train.py` still owns `trainer.train()`.

`build_callbacks(config, ...)` is the one function train.py calls to get
its full callback list, including early stopping (`transformers`' own
`EarlyStoppingCallback` -- reimplementing that logic here would just
duplicate a well-tested library feature) when
`training.early_stopping_patience` is set.

## Plain-log fallback (ZIQRA_TRAINER_PLAIN_LOG)

`rich.live.Live`'s in-place redraw works by writing cursor-movement/
clear-line ANSI escape codes and relying on the terminal to interpret
them -- it decides whether to do this via `console.is_terminal`
(`sys.stdout.isatty()`). Confirmed on a real Kaggle T4x2 run: `!python -m
...`'s subprocess stdout reports `isatty() == True` there (Kaggle's shell-
escape allocates a pty), so Rich attempts cursor-based redraw -- but
Kaggle's notebook output panel doesn't render those escape codes, so every
~0.25s refresh (`refresh_per_second=4`) appears as a brand new panel
instead of overwriting the last one, producing a wall of duplicated
panels. `is_terminal` alone can't distinguish "real terminal" from "reports
isatty()=True but the consumer won't honor cursor movement", so this can't
be auto-detected reliably -- `ZIQRA_TRAINER_PLAIN_LOG=1` is an explicit
opt-out: `RichTrainingCallback` then skips `Live`/`Progress` entirely and
prints one plain line per already-throttled event (`on_log` at
`logging_steps` cadence, `on_evaluate`, `on_save`, `on_epoch_end`) instead
of continuously redrawing a panel -- inherently spam-free since it relies
on Trainer's own logging cadence, not a fixed-rate refresh loop, to decide
when to print. Local dev and Colab are unaffected (the env var is unset
there); see train_on_kaggle.ipynb's Cell 6 for where Kaggle sets it.
"""

import logging
import os
import time
from typing import List, Optional

import torch
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from . import TrainingConfig
from .metrics import perplexity_from_loss

logger = logging.getLogger(__name__)


def _gpu_info() -> str:
    """One-line GPU/memory summary. Never raises -- reports "CPU only" if
    no accelerator is available, since this is a display, not a
    requirement."""
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        allocated = torch.cuda.memory_allocated(index) / 1e9
        reserved = torch.cuda.memory_reserved(index) / 1e9
        return f"{name} | {allocated:.2f} GB allocated / {reserved:.2f} GB reserved"
    if torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory() / 1e9
            return f"Apple MPS | {allocated:.2f} GB allocated"
        except (AttributeError, RuntimeError):
            return "Apple MPS (memory stats unavailable on this torch version)"
    return "CPU only (no GPU detected)"


class RichTrainingCallback(TrainerCallback):
    """Rich-backed terminal UI for one `trainer.train()` run -- a
    continuously redrawn Live panel by default, or a plain scrolling log
    (one line per already-throttled Trainer event) when
    `ZIQRA_TRAINER_PLAIN_LOG=1` is set. See this module's docstring for
    why the fallback exists and can't be auto-detected.

    Holds only display state (last-seen loss/LR/checkpoint/etc.) -- it
    never influences training itself. Trainer's own `TrainerState` is the
    source of truth for step/epoch/best-checkpoint; this class just
    renders it.
    """

    def __init__(self, total_epochs: int, coach: str, model_name: str) -> None:
        self._total_epochs = total_epochs
        self._coach = coach
        self._model_name = model_name

        self._console = Console()
        self._use_live = os.environ.get("ZIQRA_TRAINER_PLAIN_LOG", "").strip().lower() not in (
            "1", "true", "yes",
        )
        self._progress: Optional[Progress] = None
        self._live: Optional[Live] = None
        self._task_id = None
        self._start_time: Optional[float] = None

        self._train_loss: Optional[float] = None
        self._eval_loss: Optional[float] = None
        self._learning_rate: Optional[float] = None
        self._samples_processed = 0
        self._current_checkpoint: Optional[str] = None

    # -- Trainer hooks ---------------------------------------------------
    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        self._start_time = time.time()
        if not self._use_live:
            self._console.print(
                f"[bold]Starting training[/] -- {self._model_name} ({self._coach}), "
                f"{max(state.max_steps, 1)} total steps, {self._total_epochs} epoch(s). "
                f"Plain log mode (ZIQRA_TRAINER_PLAIN_LOG=1) -- progress prints at each "
                f"logging/eval/save/epoch event instead of a continuously redrawn panel."
            )
            return
        self._progress = Progress(
            TextColumn("[bold blue]{task.fields[stage]}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("step {task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=self._console,
        )
        self._task_id = self._progress.add_task(
            "training", total=max(state.max_steps, 1), stage=self._stage_label(state)
        )
        self._live = Live(self._render(), console=self._console, refresh_per_second=4)
        self._live.start()

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        # Always updated -- on_log's plain-mode line (below) reads this even
        # though this hook fires on every step regardless of display mode.
        self._samples_processed = (
            state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        if not self._use_live:
            return  # no per-step output in plain mode -- see on_log
        if self._progress is not None:
            self._progress.update(self._task_id, completed=state.global_step, stage=self._stage_label(state))
        self._refresh()

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[dict] = None, **kwargs
    ) -> None:
        if not logs:
            return
        if "loss" in logs:
            self._train_loss = logs["loss"]
        if "learning_rate" in logs:
            self._learning_rate = logs["learning_rate"]
        if not self._use_live:
            # The one per-step-ish print in plain mode -- but only as often
            # as Trainer itself calls on_log (every `logging_steps`), not
            # every step, so this can't spam regardless of how long a step
            # takes.
            loss_str = f"{self._train_loss:.4f}" if self._train_loss is not None else "-"
            lr_str = f"{self._learning_rate:.2e}" if self._learning_rate is not None else "-"
            self._console.print(
                f"step {state.global_step}/{max(state.max_steps, 1)} "
                f"({self._stage_label(state)}) | loss={loss_str} | lr={lr_str} | "
                f"samples={self._samples_processed}"
            )
            return
        self._refresh()

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Optional[dict] = None, **kwargs
    ) -> None:
        if metrics and "eval_loss" in metrics:
            self._eval_loss = metrics["eval_loss"]
            is_best = state.best_metric is not None and self._eval_loss <= state.best_metric
            if not self._use_live:
                # Live mode already shows Val loss in the panel; plain mode
                # has no panel, so print it here every time, not just on a
                # new best (that print, below, still fires either way).
                self._console.print(f"eval_loss={self._eval_loss:.4f}")
            if is_best:
                self._console.log(f"[bold green]New best validation loss: {self._eval_loss:.4f}[/]")
        self._refresh()

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        self._current_checkpoint = f"checkpoint-{state.global_step}"
        self._console.log(f"Saved checkpoint: {self._current_checkpoint}")
        self._refresh()

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        train_loss = f"{self._train_loss:.4f}" if self._train_loss is not None else "-"
        eval_loss = f"{self._eval_loss:.4f}" if self._eval_loss is not None else "-"
        self._console.log(
            f"Epoch {int(state.epoch)}/{self._total_epochs} complete -- train_loss={train_loss}, eval_loss={eval_loss}"
        )

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        self.stop()
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        summary = Table.grid(padding=(0, 2))
        summary.add_row("Total steps:", str(state.global_step))
        summary.add_row("Elapsed:", f"{elapsed / 60:.1f} min")
        summary.add_row(
            "Best validation loss:",
            f"{state.best_metric:.4f}" if state.best_metric is not None else "-",
        )
        summary.add_row("Best checkpoint:", state.best_model_checkpoint or "-")
        self._console.print(Panel(summary, title="Training Complete", border_style="green"))

    def stop(self) -> None:
        """Stop the Live display, if it's running. Idempotent -- safe to
        call more than once, and safe to call even if training never got
        as far as `on_train_begin`.

        `on_train_end` is Trainer's *success* hook: if `trainer.train()`
        raises partway through (a crash inside a training step, for
        example), Trainer does not reliably call it, and the Live
        display's background refresh thread keeps redrawing the full
        status panel indefinitely -- interleaving badly with the
        exception traceback as it prints, independently of whatever
        actually crashed. train.py calls this explicitly in a `finally`
        block around `trainer.train()` so a crash always gets a clean
        traceback, not a wall of repeated panels."""
        if self._live is not None and self._live.is_started:
            self._live.stop()

    # -- rendering --------------------------------------------------------
    def _stage_label(self, state: TrainerState) -> str:
        epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        return f"epoch {min(epoch, self._total_epochs)}/{self._total_epochs}"

    def _render(self) -> Group:
        status = Table.grid(padding=(0, 2))
        status.add_column(justify="right", style="bold")
        status.add_column()
        status.add_row("Model:", self._model_name)
        status.add_row("Coach:", self._coach)
        status.add_row("Train loss:", f"{self._train_loss:.4f}" if self._train_loss is not None else "-")
        status.add_row("Val loss:", f"{self._eval_loss:.4f}" if self._eval_loss is not None else "-")
        status.add_row(
            "Perplexity:",
            f"{perplexity_from_loss(self._eval_loss):.2f}" if self._eval_loss is not None else "-",
        )
        status.add_row("Learning rate:", f"{self._learning_rate:.2e}" if self._learning_rate is not None else "-")
        status.add_row("Samples processed:", str(self._samples_processed))
        status.add_row("Checkpoint:", self._current_checkpoint or "-")
        status.add_row("GPU / Memory:", _gpu_info())

        panel = Panel(status, title="Training Status", border_style="blue")
        return Group(panel, self._progress) if self._progress is not None else Group(panel)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())


def build_early_stopping_callback(patience: Optional[int]) -> Optional[EarlyStoppingCallback]:
    """`transformers.EarlyStoppingCallback(early_stopping_patience=patience)`,
    or None if `patience` is unset -- requires
    `load_best_model_at_end=True` and `metric_for_best_model` to be set on
    `TrainingArguments` (train.py sets both) to have any effect."""
    if patience is None:
        return None
    return EarlyStoppingCallback(early_stopping_patience=patience)


def build_callbacks(config: TrainingConfig, is_main_process: bool = True) -> List[TrainerCallback]:
    """The full callback list train.py passes to `Trainer(callbacks=...)`.

    `is_main_process` gates only the Rich display: under multi-GPU
    (DDP), every process would otherwise open its own Live terminal
    display and render the same panel N times, interleaved and garbled.
    It's pure display with no bearing on the training loop's control
    flow, so it's safe to attach on rank 0 only and skip entirely
    elsewhere -- unlike early stopping (attached unconditionally, below):
    that callback sets `control.should_training_stop`, which every rank
    must agree on to keep the distributed training loop synchronized, so
    it has to run identically on every process rather than being
    rank-gated.
    """
    callbacks: List[TrainerCallback] = []
    if is_main_process:
        callbacks.append(RichTrainingCallback(
            total_epochs=config.training.epochs,
            coach=config.dataset.coach,
            model_name=config.model.model_name,
        ))
    early_stopping = build_early_stopping_callback(config.training.early_stopping_patience)
    if early_stopping is not None:
        callbacks.append(early_stopping)
    return callbacks


__all__ = [
    "RichTrainingCallback",
    "build_callbacks",
    "build_early_stopping_callback",
]
