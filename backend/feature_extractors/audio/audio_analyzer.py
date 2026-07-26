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

    with ThreadPoolExecutor(max_workers=4) as executor:
        logger.info("Starting Fluency, Pronunciation, and Intonation in parallel...")
        fluency_future = executor.submit(_timed, "fluency", durations, analyze_fluency, words, sentences, speech_duration)
        pronunciation_future = executor.submit(
            _timed, "pronunciation", durations, analyze_pronunciation, audio_path, transcript, words, sentences, speech_chunks
        )
        intonation_future = executor.submit(
            _timed, "intonation", durations, analyze_intonation, audio_path, transcript, words, speech_chunks
        )

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

    logger.info("Fluency, Pronunciation, MTI, Intonation complete — running Engagement...")
    engagement_start = time.time()
    engagement_report = analyze_engagement(
        fluency_report, pronunciation_report, mti_report, intonation_report, total_duration=total_duration
    )
    durations["engagement"] = round(time.time() - engagement_start, 3)

    logger.info(f"Audio analysis complete. Analyzer durations (s): {durations}")

    return {
        "fluency": fluency_report,
        "pronunciation": pronunciation_report,
        "mti": mti_report,
        "intonation": intonation_report,
        "engagement": engagement_report,
        "processing_metadata": {
            "analyzer_durations_seconds": durations,
            "parallel_execution": True,
        },
    }
