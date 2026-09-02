"""
Scores fitted to human ratings, instead of thresholds picked by hand.

What this replaces
------------------
Every headline number in this pipeline used to be a hand-weighted blend of
hand-thresholded heuristics: pronunciation was 0.75 x phoneme-match +
0.25 x an admittedly-uncalibrated rhythm heuristic; engagement was
0.30 energy + 0.25 pause + 0.35 dynamics + 0.10 clarity minus a coverage
penalty; "monotone" meant a pitch coefficient of variation under 0.04.
Nothing in that chain had ever been compared against a human judgement, and
measurement showed what that costs — read-aloud Harvard sentences, about the
least expressive speech there is, scored 87-96 "highly engaging", and a clean
3.8s clip and a 24s accented one both landed on pronunciation 96.

These scores are regressions fitted on **speechocean762**: 2,500 training and
2,500 held-out test utterances, each rated 1-10 by expert annotators for
accuracy, fluency, prosody and completeness. The features are the same
quantities the pipeline already computes (GOP over the CTC log-probs, Praat
pitch and energy statistics, rate), so nothing new runs at request time — the
difference is that the mapping from those numbers to a score is now fitted
rather than invented.

Held-out test-split correlations are recorded in the bundle itself
(`report`) and printed by tools/fit_scorers.py, so the number quoted anywhere
is always the one measured against humans on data the model never saw.

The heuristic scores are still computed and reported under their own keys, so
the two can be compared on live traffic.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

BUNDLE_PATH = Path(__file__).with_name("models") / "scorers.joblib"

# Order is load-bearing: it is the column order the models were fitted on.
# tools/fit_scorers.py and modal_fit_scorers.py both import this list rather
# than restating it, so training and inference cannot drift apart.
FEATURE_NAMES = [
    "gop_mean", "gop_std", "gop_min", "gop_p10", "gop_p25", "gop_median",
    "gop_frac_below_1", "gop_frac_below_2", "gop_count",
    "f0_mean", "f0_std", "f0_cv", "f0_range",
    "energy_mean", "energy_std", "energy_cv",
    "voiced_fraction", "flat_window_fraction",
    "duration", "phonemes_per_second", "words_per_second", "words",
]

TARGETS = ["accuracy", "fluency", "prosodic", "completeness", "total"]

# Features the fitted models deliberately do NOT use.
#
# speechocean762 utterances are short read prompts — most a few seconds, none
# over ~20s — while an interview answer runs 15 to 120 seconds. Two problems
# follow from letting the models use raw length:
#
#   * It is out of range. A tree fitted on durations of 1.6-20s cannot say
#     anything about 90s except "same as 20s"; a linear model extrapolates,
#     which is worse.
#   * On this corpus, `duration` came out as the TOP feature for fluency,
#     prosody and total — which mostly means it had learned how long each
#     prompt takes to read. That is a property of the prompt, not of the
#     speaker, and there are no prompts in a mock interview.
#
# The cost is measured, not assumed: held-out PCC drops about 0.04-0.06
# (accuracy .724->.683, fluency .786->.721, prosodic .777->.725,
# total .748->.708). The rate features (words_per_second,
# phonemes_per_second) carry the same information length-invariantly.
#
# Worth recording what this is NOT evidence for: a 24s recording scores lower
# than a 3.8s excerpt of the same speaker, and that survived the change. It
# is not a length artefact — the excerpt is the easy opening sentence, and
# GOP says so directly (mean -0.15 with 1 weak phoneme, against -0.51 with 14
# over the full recording). The score is tracking the audio, not the clock.
LENGTH_DEPENDENT_FEATURES = ("duration", "words", "gop_count")
SCORING_FEATURES = [n for n in FEATURE_NAMES if n not in LENGTH_DEPENDENT_FEATURES]

# The pitch window monotonicity.py flags by hand; handed to the fit as a raw
# fraction instead of a verdict.
FLAT_WINDOW_SECONDS = 3.0
FLAT_CV_THRESHOLD = 0.04

_bundle = None
_load_failed = False


def _get_bundle():
    """Loads the fitted models once. A missing bundle is not fatal: the
    pipeline keeps reporting its heuristic scores and simply omits these."""
    global _bundle, _load_failed
    if _bundle is None and not _load_failed:
        try:
            import joblib

            _bundle = joblib.load(BUNDLE_PATH)
            logger.info("Loaded fitted scorers (%s).", _bundle.get("report", {}))
        except Exception as err:
            _load_failed = True
            logger.warning("Fitted scorers unavailable (%s); heuristic scores only.", err)
    return _bundle


def flat_window_fraction(f0: np.ndarray, voiced: np.ndarray, sample_rate: int, hop_length: int) -> float:
    """Fraction of 3s pitch windows whose voiced F0 is nearly flat."""
    window = max(1, int(FLAT_WINDOW_SECONDS * sample_rate / hop_length))
    step = max(1, window // 3)
    cvs: List[float] = []
    for start in range(0, max(1, len(f0) - window), step):
        segment = f0[start : start + window][voiced[start : start + window]]
        if segment.size >= 3 and segment.mean() > 0:
            cvs.append(float(segment.std() / segment.mean()))
    return float(np.mean([c < FLAT_CV_THRESHOLD for c in cvs])) if cvs else 0.0


def build_feature_vector(
    gop_scores: Sequence[Optional[float]],
    f0: np.ndarray,
    voiced: np.ndarray,
    rms: np.ndarray,
    expected_phonemes: int,
    duration_seconds: float,
    word_count: int,
    sample_rate: int,
    hop_length: int,
) -> Optional[np.ndarray]:
    """The 22 features the scorers were fitted on, or None if GOP is missing."""
    gop = np.array([g for g in gop_scores if g is not None], dtype=np.float64)
    if gop.size == 0:
        return None

    voiced_f0 = f0[voiced] if np.any(voiced) else np.array([0.0])
    duration = max(float(duration_seconds), 0.01)
    words = max(int(word_count), 1)

    return np.array([
        gop.mean(), gop.std(), gop.min(),
        np.percentile(gop, 10), np.percentile(gop, 25), np.median(gop),
        (gop < -1.0).mean(), (gop < -2.0).mean(), float(gop.size),
        voiced_f0.mean(), voiced_f0.std(),
        voiced_f0.std() / voiced_f0.mean() if voiced_f0.mean() > 0 else 0.0,
        voiced_f0.max() - voiced_f0.min(),
        rms.mean(), rms.std(),
        rms.std() / rms.mean() if rms.mean() > 0 else 0.0,
        float(np.mean(voiced)), flat_window_fraction(f0, voiced, sample_rate, hop_length),
        duration, expected_phonemes / duration, words / duration, float(words),
    ], dtype=np.float64)


def score(features: Optional[np.ndarray]) -> Dict[str, Optional[int]]:
    """
    Fitted scores on 0-100 (the annotators' 1-10 scale x 10), plus the
    held-out correlation each one was measured at, so a consumer can see how
    much to trust each number.
    """
    empty = {name: None for name in TARGETS}
    bundle = _get_bundle()
    if bundle is None or features is None:
        return {**empty, "fitted": False}

    # The bundle records which columns it was fitted on, so a feature added to
    # FEATURE_NAMES later cannot silently shift the columns underneath a model
    # that never saw it.
    columns = bundle.get("columns")
    if columns:
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        try:
            features = features[[index[name] for name in columns]]
        except KeyError as err:
            logger.warning("Fitted bundle expects unknown feature %s; skipping fitted scores.", err)
            return {**empty, "fitted": False}

    row = features.reshape(1, -1)
    out: Dict[str, Optional[int]] = {}
    for name, model in bundle["models"].items():
        try:
            raw = float(model.predict(row)[0])
        except Exception as err:
            logger.info("Scorer %s failed (%s).", name, err)
            out[name] = None
            continue
        out[name] = int(round(10 * min(10.0, max(1.0, raw))))
    return {**empty, **out, "fitted": True, "test_pcc": {k: v["test_pcc"] for k, v in bundle["report"].items()}}
