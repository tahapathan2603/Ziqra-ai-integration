"""
Audio analysis orchestrator: runs Fluency, Pronunciation, MTI, Intonation, and
Engagement and assembles their outputs into one report — the "analysis"
payload of the audio dataset's Level 2 (Feature Extraction) layer.

Runs the independent analyzers concurrently via ThreadPoolExecutor. This is a
deliberate, justified use of threads despite Python's GIL: the actual
expensive work here (PyTorch's wav2vec2 forward pass in Pronunciation,
NumPy/SciPy's librosa calls in Intonation) executes in C extensions that
release the GIL, so these threads do get real wall-clock concurrency, not
just the illusion of it.

Dependency graph (NOT a flat "everything in parallel" — see below):

    {Fluency, Pronunciation, Intonation}  -- parallel --\\
                                                          --> MTI --\\
                                                                     --> Engagement

MTI is deliberately NOT in the initial parallel batch. mti_analyzer.py is
designed to reuse Pronunciation's already-computed phoneme_errors/stress_errors
specifically to avoid a second, expensive wav2vec2 pass over the same audio
(see that module's own docstring) — running MTI fully in parallel with
Pronunciation would force a choice between paying for that redundant pass or
having MTI block on Pronunciation anyway, so it's scheduled as an explicit
follow-up task the instant Pronunciation resolves, not lumped into the first
batch. This also means Pronunciation is the only one of the three initial
parallel tasks touching the wav2vec2 model, so there's no GPU/MPS contention
between concurrently-running analyzers either.

Engagement is pure synthesis over the other four (no audio/model work of its
own — see engagement_analyzer.py), so it runs synchronously after everything
else, once all four inputs are available.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .engagement.engagement_analyzer import analyze_engagement
from .intonation.intonation_analyzer import analyze_intonation
from .mti.mti_analyzer import analyze_mti
from .pronunciation.pronunciation_analyzer import analyze_pronunciation
from ..fluency.fluency_analyzer import analyze_fluency

logger = logging.getLogger(__name__)


def _fitted_scores(pronunciation_report, intonation_report, fluency_report, words, total_duration):
    """
    Runs the scorers fitted on speechocean762's human ratings.

    Everything it needs was computed by the analyzers above — GOP from the
    pronunciation pass, the pitch and energy arrays from intonation — so this
    adds a feature assembly and five small regressions, not another model.
    """
    try:
        from .intonation.pitch_variation import HOP_LENGTH
        from .intonation.intonation_analyzer import SAMPLE_RATE
        from . import scoring

        arrays = intonation_report.get("_arrays") or {}
        if arrays.get("f0") is None or arrays.get("rms") is None:
            return {"fitted": False}

        features = scoring.build_feature_vector(
            gop_scores=pronunciation_report.get("_gop_scores") or [],
            f0=arrays["f0"],
            voiced=arrays["voiced"],
            rms=arrays["rms"],
            expected_phonemes=pronunciation_report.get("_expected_phonemes") or 0,
            duration_seconds=total_duration or 0.0,
            word_count=len(words or []),
            sample_rate=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
        )
        return scoring.score(features)
    except Exception as err:
        logger.info("Fitted scoring unavailable (%s); heuristic scores stand.", err)
        return {"fitted": False}


def _apply_fitted_scores(fitted, pronunciation_report, intonation_report, fluency_report, engagement_report):
    """
    Swap the headline numbers for the fitted ones, keeping the heuristics
    beside them under *_heuristic.

    Each mapping is to what the annotators were actually asked to rate:

      accuracy -> pronunciation_score   phoneme-level correctness
      prosodic -> intonation_score      prosody / delivery
      fluency  -> fluency_score         fluency (there was no single number)
      total    -> overall_score         the raters' overall impression

    `total` deliberately does NOT become engagement_score. speechocean762
    rates pronunciation, not how engaging someone is in an interview, and
    renaming one to the other would repeat exactly the mistake that kept
    engagement uncalibrated in the first place. engagement_score stays the
    heuristic it always was, now labelled as such, and the app shows the
    fitted four instead.

    `completeness` is not published at all: in the corpus it means "how much
    of the given prompt did the speaker read", and an interview answer has no
    prompt to be complete against — the fitted model for it can only be
    reading duration and word count.
    """
    for report, key, target in (
        (pronunciation_report, "pronunciation_score", "accuracy"),
        (intonation_report, "intonation_score", "prosodic"),
        (fluency_report, "fluency_score", "fluency"),
    ):
        value = fitted.get(target)
        if value is None:
            continue
        if key in report:
            report[f"{key}_heuristic"] = report[key]
        report[key] = value

    if fitted.get("total") is not None:
        pronunciation_report["overall_score"] = fitted["total"]

    # Mark the one number that is still a hand-weighted blend, so nothing
    # downstream mistakes it for a fitted score.
    if "engagement_score" in engagement_report:
        engagement_report["engagement_score_is_heuristic"] = True


def _vocal_arousal(audio_path: str):
    """Arousal/dominance/valence for the whole recording. Never fatal: if the
    model cannot load, engagement falls back to its heuristic score alone."""
    try:
        from .engagement.vocal_arousal import analyze_vocal_arousal
        from .intonation.intonation_analyzer import SAMPLE_RATE, _load_waveform

        return analyze_vocal_arousal(_load_waveform(audio_path), SAMPLE_RATE)
    except Exception as err:
        logger.info("Vocal arousal unavailable (%s); engagement will use heuristics only.", err)
        return {"arousal": None, "dominance": None, "valence": None, "windows": 0}


def _audio_quality(audio_path: str):
    """Channel quality, so a noisy recording is reported as one instead of
    being scored as poor speaking."""
    try:
        from .intonation.intonation_analyzer import SAMPLE_RATE, _load_waveform
        from .quality import assess_quality

        return assess_quality(_load_waveform(audio_path), SAMPLE_RATE)
    except Exception as err:
        logger.info("Quality assessment unavailable (%s).", err)
        return {"stoi": None, "pesq": None, "si_sdr": None, "usable": True, "verdict": "unknown"}


def _timed(name: str, durations: Dict[str, float], fn, *args, **kwargs):
    """Run fn and record its wall-clock duration under `name` in `durations`
    (a dict shared across threads — dict item assignment is atomic in
    CPython, so no lock is needed for this specific pattern)."""
    start = time.time()
    result = fn(*args, **kwargs)
    durations[name] = round(time.time() - start, 3)
    return result


def analyze_audio(
    audio_path: str,
    transcript: str,
    words: List[Dict],
    sentences: List[Dict],
    speech_duration: float,
    total_duration: Optional[float] = None,
    speech_chunks: Optional[List[Dict]] = None,
) -> Dict:
    """
    Run the full audio analysis pipeline (Fluency, Pronunciation, MTI,
    Intonation, Engagement), parallelized where real dependencies allow.

    Args:
        audio_path: path to the interview audio.
        transcript: full transcript text.
        words: [{"word": str, "start": float, "end": float, ...}, ...]
        sentences: [{"text": str, "start": float, "end": float}, ...]
        speech_duration: total speech time in seconds (fluency's per-minute rates).
        total_duration: full recording duration in seconds, if the caller has
            it (e.g. from VAD) — passed through to Engagement's timeline.
            Estimated internally if omitted (see engagement_analyzer.py).
        speech_chunks: VAD speech regions ([{"start": float, "end": float}, ...],
            Level 1 ground truth) — passed to Intonation so silence is never
            scored as a delivery problem (Fix 4, audio-pipeline correctness-fix
            plan). Optional; falls back to unfiltered behavior if omitted.

    Returns:
        {
            "fluency": {...},
            "pronunciation": {...},
            "mti": {...},
            "intonation": {...},
            "engagement": {...},
            "processing_metadata": {
                "analyzer_durations_seconds": {"fluency": float, "pronunciation": float,
                    "mti": float, "intonation": float, "engagement": float},
                "parallel_execution": true,
            },
        }
    """
    durations: Dict[str, float] = {}

    # Six independent tasks now: fluency, pronunciation, intonation, vocal
    # arousal, audio quality, and MTI once pronunciation lands.
    with ThreadPoolExecutor(max_workers=6) as executor:
        logger.info("Starting Fluency, Pronunciation, and Intonation in parallel...")
        fluency_future = executor.submit(_timed, "fluency", durations, analyze_fluency, words, sentences, speech_duration)
        pronunciation_future = executor.submit(
            _timed, "pronunciation", durations, analyze_pronunciation, audio_path, transcript, words, sentences, speech_chunks
        )
        intonation_future = executor.submit(
            _timed, "intonation", durations, analyze_intonation, audio_path, transcript, words, speech_chunks
        )
        # Vocal arousal and audio quality are independent of everything else,
        # so they ride alongside rather than adding to the critical path.
        arousal_future = executor.submit(_timed, "vocal_arousal", durations, _vocal_arousal, audio_path)
        quality_future = executor.submit(_timed, "audio_quality", durations, _audio_quality, audio_path)

        # MTI can't start until Pronunciation's phoneme/stress data exists —
        # submit it as its own follow-up task the moment that's available,
        # rather than waiting for Intonation/Fluency too.
        pronunciation_report = pronunciation_future.result()
        logger.info("Pronunciation complete — starting MTI (reuses its phoneme/stress data)...")
        mti_future = executor.submit(
            _timed,
            "mti",
            durations,
            analyze_mti,
            audio_path,
            transcript,
            words,
            sentences=sentences,
            phoneme_errors=pronunciation_report["phoneme_errors"],
            stress_errors=pronunciation_report["stress_errors"],
        )

        fluency_report = fluency_future.result()
        intonation_report = intonation_future.result()
        mti_report = mti_future.result()

        arousal_report = arousal_future.result()
        quality_report = quality_future.result()

    logger.info("Fluency, Pronunciation, MTI, Intonation complete — running Engagement...")
    engagement_start = time.time()
    engagement_report = analyze_engagement(
        fluency_report,
        pronunciation_report,
        mti_report,
        intonation_report,
        total_duration=total_duration,
        vocal_arousal=arousal_report,
    )
    durations["engagement"] = round(time.time() - engagement_start, 3)

    # Fitted scores replace the hand-weighted composites as the headline
    # numbers (see scoring.py). The heuristics are kept beside them under
    # *_heuristic so the two can be compared on live traffic, and so nothing
    # is lost if the fitted bundle is ever unavailable.
    fitted_start = time.time()
    fitted = _fitted_scores(pronunciation_report, intonation_report, fluency_report, words, total_duration)
    if fitted.get("fitted"):
        _apply_fitted_scores(fitted, pronunciation_report, intonation_report, fluency_report, engagement_report)
    durations["fitted_scoring"] = round(time.time() - fitted_start, 3)

    # Internal hand-offs, never published.
    for report, key in ((pronunciation_report, "_gop_scores"), (pronunciation_report, "_expected_phonemes"),
                        (intonation_report, "_arrays")):
        report.pop(key, None)

    logger.info(f"Audio analysis complete. Analyzer durations (s): {durations}")

    return {
        "fluency": fluency_report,
        "pronunciation": pronunciation_report,
        "mti": mti_report,
        "intonation": intonation_report,
        "engagement": engagement_report,
        "audio_quality": quality_report,
        "fitted_scores": fitted,
        "processing_metadata": {
            "analyzer_durations_seconds": durations,
            "parallel_execution": True,
        },
    }
