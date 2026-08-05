"""
Validation metrics for student training (Part 10).

`transformers.Trainer` already computes validation loss correctly and
efficiently during `trainer.evaluate()` (it never needs the full
vocab-sized logits held in memory to do this). The only thing added here
is `perplexity_from_loss`, a pure one-line function applied to that
already-computed loss -- recomputing cross-entropy from raw logits in a
`compute_metrics` hook would duplicate that work and risk holding a
[batch, seq_len, vocab_size] tensor in memory for the whole eval set, so
this module deliberately does not do that.

Designed to grow: `build_validation_metrics` is the one function
callbacks.py/train.py call; each additional metric (rubric agreement,
coach similarity, JSON validity, reasoning similarity) is planned as its
own small function below, so adding one never requires touching the
others. None of those are implemented yet -- they require decoding
generated text and parsing the coach_output JSON schema, which belongs to
the (not-yet-built) evaluation module, not the training loop.
"""

import math
from typing import Dict

_MAX_LOSS_FOR_PERPLEXITY = 20.0  # exp(20) ~ 4.9e8 -- caps overflow on a badly-diverged model


def perplexity_from_loss(loss: float) -> float:
    """`exp(loss)`, the standard perplexity definition for a mean
    token-level cross-entropy loss. Capped at `_MAX_LOSS_FOR_PERPLEXITY`
    so an early, badly-diverged loss doesn't overflow to `inf`."""
    if loss != loss:  # NaN check without importing math.isnan for one call site
        return float("nan")
    return math.exp(min(loss, _MAX_LOSS_FOR_PERPLEXITY))


def build_validation_metrics(eval_loss: float) -> Dict[str, float]:
    """The metrics dict callbacks.py displays after each evaluation call.

    Args:
        eval_loss: `trainer.evaluate()`'s own reported `"eval_loss"` (or
            the value in a `TrainerCallback.on_evaluate` `metrics` dict).
    """
    return {
        "eval_loss": eval_loss,
        "perplexity": perplexity_from_loss(eval_loss),
    }


# ---------------------------------------------------------------------------
# Not yet implemented -- evaluation-module territory (explicitly out of
# scope for this phase). Each stub documents what it will need so a future
# addition doesn't require redesigning this module.
# ---------------------------------------------------------------------------
def rubric_agreement(*args, **kwargs):
    """Planned: decode the model's generated `scores` JSON and compare
    each rubric's predicted score against the teacher's (e.g. exact-match
    or within-1-band accuracy). Needs generated text, not eval-time
    logits -- belongs to the evaluation module."""
    raise NotImplementedError("rubric_agreement requires generation, not eval-time loss; see module docstring.")


def coach_similarity(*args, **kwargs):
    """Planned: semantic similarity between the model's generated
    coach_output and the teacher's (e.g. embedding cosine similarity per
    field). Needs generation + an embedding model."""
    raise NotImplementedError("coach_similarity requires generation + an embedding model; see module docstring.")


def json_validity(*args, **kwargs):
    """Planned: fraction of generated responses that parse as valid JSON
    matching the {scores, coach_output, reasoning_trace} contract. Needs
    generation, not eval-time logits."""
    raise NotImplementedError("json_validity requires generation; see module docstring.")


def reasoning_similarity(*args, **kwargs):
    """Planned: similarity between the model's generated reasoning_trace
    and the teacher's (e.g. ROUGE/BLEU or embedding similarity). Needs
    generation."""
    raise NotImplementedError("reasoning_similarity requires generation; see module docstring.")


__all__ = [
    "build_validation_metrics",
    "coach_similarity",
    "json_validity",
    "perplexity_from_loss",
    "reasoning_similarity",
    "rubric_agreement",
]
