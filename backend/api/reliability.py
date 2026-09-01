"""
Suppress metrics the recording is too short to support.

Every analyzer here answers with a number whether or not there was enough
audio to justify one, and the failure mode is silent and flattering: with no
measurable monotone stretch, monotonicity reports a clean 100; with no
low-energy section, energy reports 100; a rate is a division, so four seconds
of speech yields a confident "83 words per minute". Measured on real clips, a
4.5-second answer came back 96 intonation / 96 engagement / "highly engaging",
which is not a reading of the delivery — it is the absence of one.

Two reasons that matters more here than it looks. The scores are read out to
the candidate as coaching, and they are fed to an LLM as evidence, which will
happily build a paragraph of praise on a number that means "nothing was
measured".

So: below the thresholds a metric needs, it is reported as null with the
reason recorded under `reliability`, rather than as a number. Consumers that
already skip nulls (the Workers backend's compactAudioParams does) then say
nothing about delivery instead of saying something false.

Thresholds are set from what each measurement actually needs, not tuned
against a target score:

  * The pitch family works on 3-second windows stepped by 1s
    (monotonicity.WINDOW_SECONDS/STEP_SECONDS), and flags a monotone stretch
    only at MIN_MONOTONE_SECTION_SECONDS = 3s. Under ~8s of speech there are
    too few windows for "no flat stretch found" to mean anything.
  * A per-minute rate extrapolated from a few seconds is dominated by where
    the clip happens to start and stop; 15 words and 8 seconds is the point
    where words-per-minute stops moving wildly with one extra word.
  * Phoneme accuracy is a ratio over aligned phonemes, so it needs enough
    words to be a rate rather than a coin flip.

Counts (filler_count) are exempt: a count of what was heard is honest at any
length; it is only the per-minute version that extrapolates.
"""

from typing import Any, Dict, List, Optional, Tuple

# (metric key, family) -> the families are gated together because they are
# computed from the same underlying evidence.
PITCH_FAMILY = (
    "intonation_score",
    "monotonicity_score",
    "pitch_score",
    "energy_score",
    "emphasis_score",
    "dynamics_score",
    "pause_engagement_score",
    "energy_engagement_score",
    "engagement_score",
    "engagement_level",
    "delivery_label",
)
RATE_FAMILY = ("words_per_minute", "sentences_per_minute", "fillers_per_minute")
PRONUNCIATION_FAMILY = ("pronunciation_score", "phoneme_accuracy", "rhythm_score")

MIN_SPEECH_SECONDS_FOR_PITCH = 8.0
MIN_SPEECH_SECONDS_FOR_RATE = 8.0
MIN_WORDS_FOR_RATE = 15
MIN_WORDS_FOR_PRONUNCIATION = 8


def _suppress(node: Any, keys: Tuple[str, ...]) -> int:
    """Null out `keys` anywhere in a nested dict/list. Returns how many were hit."""
    hits = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if key in keys and isinstance(value, (int, float, str)):
                node[key] = None
                hits += 1
            else:
                hits += _suppress(value, keys)
    elif isinstance(node, list):
        for item in node:
            hits += _suppress(item, keys)
    return hits


def apply(level2: Dict, speech_seconds: float, word_count: int) -> Dict:
    """
    Blank out unsupported metrics in `level2` (in place) and return a
    `reliability` block describing what was suppressed and why.
    """
    reasons: List[Dict[str, Any]] = []
    analysis = level2.get("analysis", {})

    # A noisy or muffled channel degrades phoneme matching first and worst, and
    # the failure looks exactly like poor pronunciation. If the recording
    # itself is the problem, say so instead of scoring the speaker for it.
    quality = level2.get("audio_quality") or {}
    if quality.get("verdict") == "poor":
        if _suppress(analysis, PRONUNCIATION_FAMILY):
            reasons.append({
                "metrics": list(PRONUNCIATION_FAMILY),
                "reason": (
                    f"recording quality is poor (STOI {quality.get('stoi')}, PESQ {quality.get('pesq')}); "
                    "phoneme scores measure the channel as much as the speaker at this level"
                ),
            })

    if speech_seconds < MIN_SPEECH_SECONDS_FOR_PITCH:
        if _suppress(analysis, PITCH_FAMILY):
            reasons.append({
                "metrics": list(PITCH_FAMILY),
                "reason": (
                    f"only {speech_seconds:.1f}s of speech; pitch and energy measures need at least "
                    f"{MIN_SPEECH_SECONDS_FOR_PITCH:.0f}s to have enough windows to mean anything"
                ),
            })

    if speech_seconds < MIN_SPEECH_SECONDS_FOR_RATE or word_count < MIN_WORDS_FOR_RATE:
        if _suppress(analysis, RATE_FAMILY):
            reasons.append({
                "metrics": list(RATE_FAMILY),
                "reason": (
                    f"{word_count} words in {speech_seconds:.1f}s; a per-minute rate extrapolated from "
                    f"less than {MIN_WORDS_FOR_RATE} words or {MIN_SPEECH_SECONDS_FOR_RATE:.0f}s is noise"
                ),
            })

    if word_count < MIN_WORDS_FOR_PRONUNCIATION:
        if _suppress(analysis, PRONUNCIATION_FAMILY):
            reasons.append({
                "metrics": list(PRONUNCIATION_FAMILY),
                "reason": f"only {word_count} words; too few aligned phonemes for an accuracy ratio",
            })

    return {
        "speech_seconds": round(speech_seconds, 2),
        "word_count": word_count,
        "suppressed": reasons,
        "fully_measured": not reasons,
    }


def rescale_rhythm(level2: Dict) -> None:
    """
    Put rhythm_score on the same 0-100 scale as every other exposed score.

    It is computed 0-1 internally, and was published that way alongside a set
    of 0-100 metrics — so a consumer (and the coaching LLM reading these as a
    list) saw "rhythm score 0.77" next to "pronunciation score 38" and had no
    way to know they were different scales.
    """
    pronunciation = level2.get("analysis", {}).get("pronunciation")
    if isinstance(pronunciation, dict):
        value = pronunciation.get("rhythm_score")
        if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
            pronunciation["rhythm_score"] = round(value * 100)
