"""
Vocal arousal, dominance and valence from audeering's MSP-Podcast model.

Why a model here
----------------
engagement_score was a weighted blend of four hand-written heuristics (energy
patterns, pause patterns, speaking dynamics, clarity) with hand-picked
weights and no calibration against anything. Measured on real recordings it
does not discriminate: read-aloud Harvard sentences — about the least
expressive speech there is — scored 87 to 96, "highly engaging", and a 4.5s
clip scored 96 because the analyzers found no problem sections to subtract
for.

`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` predicts arousal,
dominance and valence, fine-tuned on MSP-Podcast — the largest naturalistic
emotional-speech corpus — reaching a concordance correlation around 0.76-0.82
for arousal. Arousal is not the same thing as "engaging in an interview", and
this module does not pretend otherwise; it is, however, a measurement of
vocal activation made against human ratings, which is more than any of the
four heuristics can claim.

How it is used
--------------
Reported in full, and blended half-and-half with the existing composite (see
engagement_analyzer). Both sides stay visible in the report so the blend is
auditable and can be replaced the moment there is labelled interview data to
fit against — the alternative, picking a fifth weight for a five-way blend,
would be inventing precision again.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

logger = logging.getLogger(__name__)

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# The model mean-pools over whatever it is given, so a two-minute answer would
# collapse to one number dominated by nothing in particular. Windowing keeps
# each estimate over a stretch short enough to be about one delivery, and
# bounds peak memory on long answers.
WINDOW_SECONDS = 12.0
# Below this there is not enough voice to estimate activation from.
MIN_SECONDS = 1.5

_processor: Optional[Wav2Vec2Processor] = None
_model: Optional["EmotionModel"] = None
_device: Optional[torch.device] = None


class RegressionHead(nn.Module):
    """The checkpoint's own head — this architecture is not in transformers,
    so it has to be declared here to load the weights (as the model card does)."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden, dim=1)
        return pooled, self.classifier(pooled)


def _get_model():
    global _processor, _model, _device
    if _model is None:
        logger.info("Loading vocal arousal model (%s)...", MODEL_NAME)
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = EmotionModel.from_pretrained(MODEL_NAME)
        _model.eval()
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        logger.info("Vocal arousal model running on %s.", _device)
    return _processor, _model, _device


def analyze_vocal_arousal(waveform: np.ndarray, sample_rate: int) -> Dict:
    """
    Returns {"arousal": 0-100, "dominance": 0-100, "valence": 0-100,
             "windows": int} — or all None when there is too little audio.

    The model's own outputs are roughly 0-1; they are put on 0-100 here to
    match every other score the pipeline publishes.
    """
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    duration = len(audio) / sample_rate
    if duration < MIN_SECONDS:
        return {"arousal": None, "dominance": None, "valence": None, "windows": 0}

    window = int(WINDOW_SECONDS * sample_rate)
    chunks: List[np.ndarray] = [audio[i : i + window] for i in range(0, len(audio), window)]
    # A trailing sliver is noise, not a window — fold it into the previous one.
    if len(chunks) > 1 and len(chunks[-1]) < MIN_SECONDS * sample_rate:
        chunks[-2] = np.concatenate([chunks[-2], chunks.pop()])

    processor, model, device = _get_model()
    predictions = []
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        values = inputs.input_values.to(device)
        with torch.no_grad():
            _, logits = model(values)
        predictions.append(logits.squeeze(0).cpu().numpy())

    # Weighted by how much audio each window actually covered, so a short
    # final window doesn't count as much as a full one.
    weights = np.array([len(c) for c in chunks], dtype=np.float64)
    mean = np.average(np.stack(predictions), axis=0, weights=weights)
    arousal, dominance, valence = (float(v) for v in mean)

    def to_score(value: float) -> int:
        return int(round(100 * min(1.0, max(0.0, value))))

    return {
        "arousal": to_score(arousal),
        "dominance": to_score(dominance),
        "valence": to_score(valence),
        "windows": len(chunks),
    }
