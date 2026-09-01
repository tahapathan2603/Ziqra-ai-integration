"""
Goodness of Pronunciation (GOP) from the CTC logits the pipeline already computes.

Why this exists
---------------
phoneme_accuracy.py scores pronunciation by string comparison: run the
wav2vec2 recogniser, take its argmax phoneme sequence, and align that against
what espeak says the words should sound like. Every position that fails to
match is an error.

That makes the score a function of whether the recogniser's single best guess
happened to be right, which is a harsher and noisier thing than whether the
candidate pronounced the word acceptably. A phoneme the model rated 0.45
against a 0.46 rival counts identically to one it rated 0.01 — both are simply
"wrong" — so the headline score jumps in steps of 1/N and reads as a
mispronunciation whenever the recogniser merely hesitated.

GOP is the standard answer in the pronunciation-assessment literature: instead
of asking "did the recogniser output this phoneme", ask "how much did the
acoustics support this phoneme, relative to the alternatives". It is graded,
it degrades smoothly, and it distinguishes a genuine substitution from a
close call. Recent work (Interspeech 2025, "Evaluating Logit-Based GOP Scores
for Mispronunciation Detection") finds logit-margin variants track human
judgement better than posterior-only ones, which is the form used here.

What it does NOT do
-------------------
It does not replace the existing accuracy number, because turning GOP into a
calibrated 0-100 score needs labelled pronunciation data (speechocean762 is
the usual source) that this repo does not have. Inventing a curve would be
guessing with more decimal places. What it does instead:

  * attaches a per-phoneme `gop` to the reported errors, so consumers can see
    how strongly the acoustics disagreed;
  * suppresses "errors" the acoustics actually support, which is a precision
    fix that needs no calibration — if the model gave the expected phoneme a
    strong score and simply preferred a near neighbour, that is a recogniser
    artefact, not a mispronunciation;
  * reports the distribution (mean, and how many phonemes fall below the
    confidence floor) so a calibrated score can be fitted later against real
    labels rather than assumed now.
"""

import logging
from typing import Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Above this, the acoustics supported the expected phoneme well enough that a
# reported mismatch is more likely a recogniser artefact than a
# mispronunciation. GOP here is a log-probability margin: 0 means the expected
# phoneme WAS the model's top choice for its frames, -1 means it was about
# e-times less likely than the winner, and so on. -1.0 is deliberately
# permissive: this gate only ever removes errors, so a loose threshold errs
# toward reporting a real problem rather than inventing one.
GOP_SUPPORTS_EXPECTED = -1.0


def phoneme_gop(
    log_probs: torch.Tensor,
    target_ids: Sequence[int],
    blank_id: int,
) -> Optional[List[float]]:
    """
    Per-phoneme GOP for one utterance.

    Args:
        log_probs: (frames, vocab) log-softmax over the CTC vocabulary.
        target_ids: the expected phoneme sequence, as vocabulary ids.
        blank_id: the CTC blank.

    Returns one score per entry in `target_ids` — the mean over that
    phoneme's aligned frames of (log P(expected) - log P(best competitor)),
    so 0.0 means "the model's own top choice", and more negative means the
    acoustics pointed somewhere else. None if alignment is not possible
    (more phonemes than frames, or an empty sequence).
    """
    if not len(target_ids) or log_probs.shape[0] < len(target_ids):
        return None

    try:
        from torchaudio.functional import forced_align
    except ImportError:  # pragma: no cover - torchaudio is a hard dep in the image
        logger.info("torchaudio.functional.forced_align unavailable; skipping GOP.")
        return None

    targets = torch.tensor([list(target_ids)], dtype=torch.int32, device=log_probs.device)
    try:
        aligned, _scores = forced_align(
            log_probs.unsqueeze(0).float(), targets, blank=blank_id
        )
    except Exception as err:  # alignment can legitimately fail on odd inputs
        logger.info("Forced alignment failed (%s); skipping GOP for this utterance.", err)
        return None

    frame_labels = aligned[0].tolist()
    best = log_probs.max(dim=-1).values

    # Walk the alignment, collecting frames per target position. forced_align
    # emits the target sequence in order with blanks interleaved, so a change
    # of non-blank label advances to the next expected phoneme.
    sums: List[float] = [0.0] * len(target_ids)
    counts: List[int] = [0] * len(target_ids)
    position = -1
    previous = None
    for frame, label in enumerate(frame_labels):
        if label == blank_id:
            previous = label
            continue
        if label != previous:
            position += 1
        previous = label
        if 0 <= position < len(target_ids):
            sums[position] += float(log_probs[frame, target_ids[position]] - best[frame])
            counts[position] += 1

    return [round(sums[i] / counts[i], 3) if counts[i] else None for i in range(len(target_ids))]


def summarise(gop_scores: Sequence[Optional[float]]) -> Dict:
    """Distribution summary for the report — see the module docstring on why
    this is reported rather than folded into the headline score."""
    scored = [g for g in gop_scores if g is not None]
    if not scored:
        return {"mean_gop": None, "weak_phonemes": None, "scored_phonemes": 0}
    weak = sum(1 for g in scored if g < GOP_SUPPORTS_EXPECTED)
    return {
        "mean_gop": round(sum(scored) / len(scored), 3),
        "weak_phonemes": weak,
        "scored_phonemes": len(scored),
    }
