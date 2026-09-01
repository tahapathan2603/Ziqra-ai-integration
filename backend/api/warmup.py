"""
Container warm-up: pay the pipeline's one-time costs before a user is waiting
on them.

Measured on an A10G with a 4.5s clip, three identical requests in one
container:

    call 1: 28.5s total  (intonation 23.5s, pronunciation 6.4s)
    call 2:  1.37s total (intonation 0.21s, pronunciation 0.33s)
    call 3:  1.36s total (intonation 0.23s, pronunciation 0.34s)

So essentially the whole 28s is one-time per container — librosa/numba JIT
compilation on the first pitch/energy call, plus loading Whisper and the
wav2vec2 phoneme model — and the actual analysis of a short answer takes about
a second and a half. Nothing about the pipeline is slow; the first request
through a cold container is.

`/health` used to return without touching any of it, which is why the app's
pre-warm ping (fired when a user starts onboarding, precisely to absorb this)
did nothing but boot a container and leave the 28s for the first real answer.
Warm-up now starts the moment the ASGI app does, in a background thread so the
server still answers /health immediately and can report whether it is ready.
"""

import logging
import threading
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_state: Dict[str, Optional[object]] = {"warm": False, "seconds": None, "error": None}
_lock = threading.Lock()
_started = False

# 2 seconds of a voiced-sounding synthetic signal: long enough for the pitch
# and energy paths to run their real code (and so compile it), short enough
# that warm-up is dominated by compilation rather than by analysing audio.
_SYNTH_SECONDS = 2.0
_SAMPLE_RATE = 16000


def _synthetic_speech() -> np.ndarray:
    """A cheap voiced-like signal: an F0 sweep with harmonics and a little noise.

    It only has to exercise the same numeric paths as real audio — yin, rms,
    the windowed statistics — not to sound like anything.
    """
    t = np.linspace(0, _SYNTH_SECONDS, int(_SAMPLE_RATE * _SYNTH_SECONDS), endpoint=False)
    f0 = 120 + 40 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / _SAMPLE_RATE
    signal = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    signal += 0.01 * np.random.default_rng(0).standard_normal(t.shape)
    return (signal * 0.5).astype(np.float32)


def _warm() -> None:
    start = time.time()
    try:
        # 1. Whisper — several seconds to load onto the GPU.
        from backend.preprocessing.speech_to_text import get_model as get_whisper

        get_whisper()

        # 2. wav2vec2 phoneme recogniser, and one tiny inference so its CUDA
        #    kernels are compiled too, not just the weights loaded.
        import torch

        from backend.feature_extractors.audio.pronunciation import phoneme_accuracy

        phoneme_accuracy._get_model()
        try:
            phoneme_accuracy._detect_phonemes_batch(
                torch.from_numpy(_synthetic_speech()), _SAMPLE_RATE, [(0.0, _SYNTH_SECONDS)]
            )
        except Exception as err:  # pragma: no cover - internal helper may change shape
            logger.info("Warm-up: phoneme probe skipped (%s)", err)

        # 3. The expensive one: librosa/numba compilation on the pitch and
        #    energy paths, plus the windowed statistics built on top of them.
        from backend.feature_extractors.audio.intonation.emphasis import analyze_emphasis
        from backend.feature_extractors.audio.intonation.energy_variation import (
            analyze_energy_variation,
            compute_energy_contour,
        )
        from backend.feature_extractors.audio.intonation.monotonicity import analyze_monotonicity
        from backend.feature_extractors.audio.intonation.pitch_variation import (
            analyze_pitch_variation,
            compute_pitch_contour,
        )

        wave = _synthetic_speech()
        pitch = compute_pitch_contour(wave, _SAMPLE_RATE)
        energy = compute_energy_contour(wave, _SAMPLE_RATE)
        analyze_pitch_variation(wave, _SAMPLE_RATE, pitch_contour=pitch)
        analyze_energy_variation(wave, _SAMPLE_RATE, energy_contour=energy, speech_chunks=None, voiced=pitch["voiced"])
        analyze_monotonicity(wave, _SAMPLE_RATE, pitch_contour=pitch, speech_chunks=None)
        analyze_emphasis([], wave, _SAMPLE_RATE, pitch_contour=pitch, energy_contour=energy)

        # 4. Vocal arousal (engagement) and the SQUIM quality model — both
        #    load weights on first use, same as the two above.
        from backend.feature_extractors.audio.engagement.vocal_arousal import analyze_vocal_arousal
        from backend.feature_extractors.audio.quality import assess_quality

        probe = _synthetic_speech()
        try:
            analyze_vocal_arousal(probe, _SAMPLE_RATE)
        except Exception as err:  # pragma: no cover
            logger.info("Warm-up: arousal model skipped (%s)", err)
        try:
            assess_quality(probe, _SAMPLE_RATE)
        except Exception as err:  # pragma: no cover
            logger.info("Warm-up: quality model skipped (%s)", err)

        # 5. espeak/phonemizer's first call also pays a library-load cost, and
        #    Praat's pitch tracker compiles nothing but does load a library.
        phoneme_accuracy.get_expected_phonemes("warmup")

        _state["warm"] = True
        _state["seconds"] = round(time.time() - start, 2)
        logger.info("Warm-up complete in %ss — models loaded, JIT compiled.", _state["seconds"])
    except Exception as err:  # never take the server down over warm-up
        _state["error"] = str(err)
        _state["seconds"] = round(time.time() - start, 2)
        logger.exception("Warm-up failed after %ss; first request will pay the cost instead.", _state["seconds"])


def start_warmup() -> None:
    """Kick warm-up off once, in the background. Safe to call repeatedly."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_warm, name="pipeline-warmup", daemon=True).start()


def status() -> Dict[str, Optional[object]]:
    """{'warm': bool, 'seconds': float|None, 'error': str|None} for /health."""
    return dict(_state)


def wait_until_warm(timeout: float) -> bool:
    """Block until warm or `timeout` elapses. Used by /health?wait=1 so a
    caller that wants a guaranteed-warm container can ask for one."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _state["warm"] or _state["error"]:
            return bool(_state["warm"])
        time.sleep(0.25)
    return bool(_state["warm"])
