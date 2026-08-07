"""
Training callbacks for student training (Part 10) -- early stopping.

`build_callbacks(config, ...)` is the one function train.py calls to get
its full callback list. Today that list holds at most one entry:
`transformers`' own `EarlyStoppingCallback` (reimplementing that logic
here would just duplicate a well-tested library feature), attached when
`training.early_stopping_patience` is set.

## Why there is no custom progress UI here

This module used to own `RichTrainingCallback`, a `rich.live.Live`
terminal display (status panel + progress bar, redrawn 4x/second). It was
removed: `Live` redraws in place by writing cursor-movement ANSI codes and
trusting the consumer to honor them, gated on `console.is_terminal`
(`sys.stdout.isatty()`). Kaggle's `!python` subprocess reports
`isatty() == True` (it allocates a pty) but its notebook output panel does
not act on cursor movement, so every refresh landed as a *new* panel --
burying real output under dozens of duplicates on a live T4x2 run. That is
not reliably detectable at runtime, so there is no correct auto-fallback to
write.

`transformers.Trainer` already ships the thing this was reimplementing: set
`disable_tqdm=False` (train.py does) and its built-in `ProgressCallback`
renders a tqdm bar with step/total/elapsed/ETA and prints each logged
loss/learning-rate dict above it. tqdm updates with carriage returns rather
than cursor addressing, which is what every Hugging Face example running on
Kaggle relies on. Deleting the custom UI removed ~170 lines and one whole
class of platform-specific rendering bugs; per-run GPU/memory reporting, the
one thing tqdm does not cover, belongs in an explicit log line rather than a
continuously repainted panel.
"""

import logging
from typing import List, Optional

from transformers import EarlyStoppingCallback, TrainerCallback

from . import TrainingConfig

logger = logging.getLogger(__name__)


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

    `is_main_process` is accepted but intentionally unused: it existed to
    gate the old Rich display (see this module's docstring) to rank 0, so
    that a multi-GPU run wouldn't open N interleaved terminal displays.
    Early stopping is the opposite case and must NOT be rank-gated -- it
    sets `control.should_training_stop`, which every rank has to agree on
    to keep the distributed training loop synchronized, so it runs
    identically on every process. The parameter is kept so that adding a
    future display/logging callback here has an obvious, already-plumbed
    place to be gated, rather than the caller having to rediscover the
    distinction.
    """
    callbacks: List[TrainerCallback] = []
    early_stopping = build_early_stopping_callback(config.training.early_stopping_patience)
    if early_stopping is not None:
        callbacks.append(early_stopping)
    return callbacks


__all__ = [
    "build_callbacks",
    "build_early_stopping_callback",
]
