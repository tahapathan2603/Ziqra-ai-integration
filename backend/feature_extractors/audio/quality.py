"""
Non-intrusive audio-quality assessment, so a bad recording is reported as one
rather than scored as bad speaking.

Everything downstream — phoneme accuracy, pitch tracking, energy — degrades on
a noisy or clipped recording, and degrades *silently*: the candidate gets told
their pronunciation is poor when what actually happened is that they were in a
cafe, or their phone mic was covered. Nothing in the pipeline could tell those
apart.

torchaudio's SQUIM_OBJECTIVE predicts STOI (intelligibility), PESQ (perceptual
quality) and SI-SDR without needing a clean reference to compare against,
which is exactly the situation here. It is already part of torchaudio, so this
costs no new dependency and a 28MB model.

Thresholds are set from measurement on this repo's own recordings rather than
from the papers' defaults — see THRESHOLDS below.
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_model = None
_device = None

# SQUIM is trained on 16kHz speech; the pipeline resamples everything to 16k
# already (silero_vad.SAMPLE_RATE).
EXPECTED_SAMPLE_RATE = 16000

# Long recordings are assessed on a sample rather than end to end: quality is a
# property of the channel, not of the sentence, and the model's memory use
# grows with length.
MAX_SECONDS = 20.0

# Measured on this repo's own recordings:
#
#   clean SAA recording        STOI 0.994   PESQ 2.88
#   same file at 5dB SNR       STOI 0.783   PESQ 1.12   <- the case to catch
#   real phone recording (opus) STOI 0.947  PESQ 1.50
#   short clean clip           STOI 0.995   PESQ 3.16
#
# STOI separates those cleanly on its own. PESQ does not: it scores perceptual
# quality, so it penalises lossy coding, and the real phone recording — which
# is perfectly intelligible — lands at 1.50, below any PESQ threshold that
# would also catch the 5dB case. Since every recording this pipeline receives
# arrives through opus or webm from a browser or phone, gating on PESQ would
# declare ordinary users' audio unusable. It stays in the report as context,
# out of the decision.
STOI_POOR = 0.85


def _get_model():
    """
    Loads SQUIM once, onto the GPU when there is one.

    Measured: left on the CPU this took 8.9s for a 20s clip — longer than
    every other analyzer combined, for a 28MB model, on a box with an A10G
    sitting idle.
    """
    global _model, _device
    if _model is None:
        from torchaudio.pipelines import SQUIM_OBJECTIVE

        logger.info("Loading SQUIM objective quality model...")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = SQUIM_OBJECTIVE.get_model().to(_device)
        _model.eval()
        logger.info("SQUIM running on %s.", _device)
    return _model, _device


def assess_quality(waveform: np.ndarray, sample_rate: int = EXPECTED_SAMPLE_RATE) -> Dict:
    """
    Returns {"stoi": float, "pesq": float, "si_sdr": float, "usable": bool,
             "verdict": str} — or Nones with verdict "unknown" if assessment
    fails, which must never take an extraction down.
    """
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if len(audio) < sample_rate:  # under a second: nothing to judge
        return _unknown("too short to assess")

    if len(audio) > MAX_SECONDS * sample_rate:
        audio = audio[: int(MAX_SECONDS * sample_rate)]

    try:
        model, device = _get_model()
        with torch.no_grad():
            stoi, pesq, si_sdr = model(torch.from_numpy(audio).unsqueeze(0).to(device))
        stoi_v, pesq_v, sdr_v = float(stoi[0]), float(pesq[0]), float(si_sdr[0])
    except Exception as err:
        logger.info("Quality assessment failed (%s); continuing without it.", err)
        return _unknown(str(err))

    poor = stoi_v < STOI_POOR
    return {
        "stoi": round(stoi_v, 3),
        "pesq": round(pesq_v, 3),
        "si_sdr": round(sdr_v, 2),
        "usable": not poor,
        "verdict": "poor" if poor else "ok",
    }


def _unknown(reason: str) -> Dict:
    return {"stoi": None, "pesq": None, "si_sdr": None, "usable": True, "verdict": "unknown", "reason": reason}
