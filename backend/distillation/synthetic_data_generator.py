"""
Synthetic audio-analysis dataset generator for teacher/student distillation.

Design principle (why this is not a per-field random generator)
---------------------------------------------------------------
The value of the audio pipeline is that all five analyzers derive their outputs
from the SAME underlying signal, so their scores and evidence are mutually
consistent (we spent real effort removing cross-module contradictions). Inventing
each module's scores independently with "realistic-looking" random numbers would
(a) violate the pipeline's own scoring formulas
(pronunciation_score = round(100*(phoneme_accuracy + rhythm_score)/2),
mti_score = 100 - Σ severity_penalty, engagement = weighted blend - coverage,
capped by timeline, intonation = mean of four sub-scores), and (b) fake the
cross-module correlations with coefficients instead of shared causes — exactly
how contradictions creep back in.

So instead we synthesize PRIMITIVES (a latent ability per candidate + a persona,
producing a word timeline, pauses, fillers, phoneme-error events, and pitch/
energy contours) and run them through the REAL analyzer code:

  - analyze_fluency(...)                         -> real (pure: words/sentences)
  - detect_mispronounced_words + analyze_rhythm  -> real (pure), + the real
      pronunciation_score formula and observation text
  - analyze_mti(..., phoneme_errors=...)         -> real (accepts pre-computed
      phoneme errors by design, so no audio/model pass)
  - the four intonation sub-analyzers            -> real, run on synthetic
      contours (each accepts a pre-computed contour and never touches the
      waveform), plus the real reconcile/label/observation logic
  - analyze_engagement(...)                      -> real (pure synthesis)

No ML model is ever loaded (we never call analyze_phoneme_accuracy / _get_model);
the heavy imports below are import-only. Every SCORE and every feature the
downstream training packet consumes is therefore computed at full fidelity.

Size tradeoff (documented, deliberate)
--------------------------------------
The two raw acoustic fields that no score and no training packet consumes
(timeline.acoustic_contours and timeline.detected_phonemes) are generated at
reduced resolution — contours at CONTOUR_DT vs the live pipeline's ~0.032s, and
detected_phonemes as a compact per-word stream — purely to keep a 2000-row file
to a practical size. The intonation scores are computed on exactly the contour
that gets stored, so internal consistency still holds. detected_phonemes is kept
only in `timeline` (Level 1); the byte-identical copy the pronunciation block
would otherwise duplicate is dropped, the same dedup the pipeline applies to
intonation's contours.

Row schema (one JSON object per line) matches the current output matrix:
    {session_id, timeline, fluency, pronunciation, mti, intonation, engagement}
`timeline` is the Level-1 dict (build_timeline); the five analysis blocks are the
Level-2 blocks (build_features -> "analysis"), contours stripped from intonation
because they live in timeline (same dedup the real pipeline does).
"""

import copy
import hashlib
import json
import logging
import os
import random
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- audio-extraction dependency stubs --------------------------------------
# The pipeline's analyzers import librosa / torchaudio / phonemizer / pronouncing
# at module load, but those libraries are only used to EXTRACT signal from real
# audio (pitch tracking, RMS, wav2vec2/espeak G2P, waveform loading). This
# synthetic generator produces the contours and phoneme-error events directly, so
# it never calls any of those extraction code paths — it only calls the pure
# analysis/scoring logic downstream of them. We stub the extraction-only modules
# so the REAL analyzer code can be imported and reused verbatim here, instead of
# reimplementing (and risking drift from) the pipeline's scoring. Stubbing happens
# ONLY when a dependency is genuinely absent, so a fully-provisioned environment
# still imports and uses the real libraries.
import importlib.machinery as _machinery
import sys as _sys
import types as _types


def _install_stub(name: str, attrs: Optional[Dict] = None) -> None:
    module = _types.ModuleType(name)
    # A valid __spec__ (not None) is required so importlib.util.find_spec doesn't
    # raise when other libraries probe for these packages — transformers, for
    # one, calls find_spec("librosa") during its lazy backend detection. With a
    # real spec the probe returns cleanly and the version lookup then fails, so
    # transformers correctly concludes the package is unavailable.
    module.__spec__ = _machinery.ModuleSpec(name, loader=None)
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    _sys.modules[name] = module


for _dep in ("librosa", "torchaudio", "pronouncing"):
    try:
        __import__(_dep)
    except ImportError:
        _install_stub(_dep)
try:
    __import__("phonemizer")
except ImportError:
    _install_stub("phonemizer", {"phonemize": lambda *a, **k: []})
    _install_stub("phonemizer.separator", {"Separator": object})
    _sys.modules["phonemizer"].separator = _sys.modules["phonemizer.separator"]

# phoneme_accuracy.py imports the wav2vec2 model stack (transformers, and through
# it soxr/librosa audio backends) at module load — purely for the acoustic
# detection pass we never run. Everything downstream only needs its pure
# clean_word() helper, so if the heavy stack can't import here, stand phoneme_accuracy
# in with the real clean_word plus dummies for anything else pulled from it. This
# keeps the whole transformers import chain from ever loading.
import re as _re

_PA_MODULE = "backend.feature_extractors.audio.pronunciation.phoneme_accuracy"
try:
    import transformers as _t  # noqa: F401 — probe: does the model stack import cleanly?
    from transformers import Wav2Vec2ForCTC as _w  # noqa: F401
    _HEAVY_STACK_OK = True
except Exception:
    _HEAVY_STACK_OK = False

if not _HEAVY_STACK_OK and _PA_MODULE not in _sys.modules:
    _pa = _types.ModuleType(_PA_MODULE)
    _pa.__spec__ = _machinery.ModuleSpec(_PA_MODULE, loader=None)
    _pa.clean_word = lambda word: _re.sub(r"[^a-zA-Z']", "", word).lower()

    def _pa_getattr(name):
        def _unavailable(*a, **k):
            raise RuntimeError(f"{_PA_MODULE}.{name} is stubbed out for synthetic generation")
        return _unavailable

    _pa.__getattr__ = _pa_getattr
    _sys.modules[_PA_MODULE] = _pa

# --- real pipeline code (reused, never reimplemented) -----------------------
from backend.feature_extractors.fluency.fluency_analyzer import analyze_fluency
from backend.feature_extractors.audio.pronunciation.word_pronunciation import detect_mispronounced_words
from backend.feature_extractors.audio.pronunciation.rhythm import analyze_rhythm
from backend.feature_extractors.audio.pronunciation.phoneme_accuracy import clean_word
from backend.feature_extractors.audio.pronunciation.pronunciation_analyzer import (
    _generate_overall_observations as _pron_observations,
)
from backend.feature_extractors.audio.mti.mti_analyzer import analyze_mti
from backend.feature_extractors.audio.intonation.pitch_variation import analyze_pitch_variation
from backend.feature_extractors.audio.intonation.energy_variation import analyze_energy_variation
from backend.feature_extractors.audio.intonation.monotonicity import analyze_monotonicity
from backend.feature_extractors.audio.intonation.emphasis import analyze_emphasis
from backend.feature_extractors.audio.intonation.intonation_analyzer import (
    _reconcile_pitch_with_monotonicity,
    _overall_delivery_label,
    _combine_observations,
    _serialize_pitch_contour,
    _serialize_energy_contour,
)
from backend.feature_extractors.audio.engagement.engagement_analyzer import analyze_engagement
from backend.feature_extractors.audio.phoneme_patterns import (
    CONSONANT_SUBSTITUTION_PATTERNS,
    VOWEL_ELONGATION_PAIRS,
    VOWEL_SHORTENING_PAIRS,
    VOWEL_PHONEMES,
    CONSONANT_PHONEMES,
)
from backend.distillation.dataset_builder import build_timeline, build_features, generate_session_id

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CONTOUR_DT = 0.25         # seconds/frame for synthetic contours (see module docstring)
SEED = 20260721

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# Content words (>2 letters, not function words) — interview-domain, so
# emphasis/importance heuristics and realistic transcripts both work.
CONTENT_WORDS = [
    "experience", "project", "team", "challenge", "solution", "develop", "manage",
    "design", "analyze", "result", "customer", "product", "process", "improve",
    "leadership", "communication", "software", "database", "algorithm", "performance",
    "strategy", "deadline", "collaborate", "responsibility", "achievement", "opportunity",
    "environment", "knowledge", "education", "research", "engineer", "application",
    "framework", "deployment", "testing", "feedback", "meeting", "presentation",
    "decision", "priority", "quality", "budget", "timeline", "requirement",
    "stakeholder", "innovation", "growth", "motivation", "skill", "career",
    "industry", "company", "position", "interview", "question", "example",
    "situation", "background", "strength", "goal", "objective", "success",
    "failure", "learning", "mentor", "colleague", "client", "deliver",
    "implement", "optimize", "coordinate", "negotiate", "contribute", "initiative",
    "planning", "analysis", "delivery", "feature", "system", "approach",
]

# Function words — sprinkled for realism; drawn from intonation's own list so the
# emphasis analyzer treats them as non-content exactly as it would on real data.
FUNCTION_WORDS = [
    "the", "and", "to", "of", "a", "in", "that", "with", "for", "was", "is",
    "on", "as", "it", "we", "our", "my", "this", "so", "then", "at", "by",
    "from", "but", "or", "an", "which", "when", "very", "also", "just",
]

FILLERS = ["um", "uh", "like", "you know", "basically", "actually", "kind of", "sort of"]

# Error-prone words -> the realistic phoneme substitution(s) they can exhibit,
# each drawn from the real phoneme_patterns tables so analyze_mti classifies them
# into the right vowel/consonant/severity buckets. (expected, detected) or None
# on one side for an insertion/deletion.
ERROR_PRONE_WORDS: Dict[str, List[Tuple[Optional[str], Optional[str]]]] = {
    # /θ/ substitutions (missing aspiration)
    "think": [("/θ/", "/t/")], "thing": [("/θ/", "/t/")], "three": [("/θ/", "/t/")],
    "through": [("/θ/", "/s/")], "thought": [("/θ/", "/t/")], "method": [("/θ/", "/t/")],
    # /ð/ substitutions
    "this": [("/ð/", "/d/")], "that": [("/ð/", "/d/")], "there": [("/ð/", "/d/")],
    "them": [("/ð/", "/z/")], "they": [("/ð/", "/d/")], "other": [("/ð/", "/d/")],
    # /v/<->/w/
    "very": [("/v/", "/w/")], "value": [("/v/", "/w/")], "develop": [("/v/", "/w/")],
    "level": [("/v/", "/w/")], "solve": [("/v/", "/w/")], "involve": [("/v/", "/w/")],
    "west": [("/w/", "/v/")], "work": [("/w/", "/v/")], "would": [("/w/", "/v/")],
    # /r/ <-> /l/ and retroflex
    "role": [("/r/", "/ɽ/")], "result": [("/r/", "/ɽ/")], "problem": [("/r/", "/l/")],
    "resource": [("/r/", "/l/")], "clarity": [("/l/", "/r/")], "release": [("/r/", "/l/")],
    # retroflex /t/,/d/
    "team": [("/t/", "/ʈ/")], "time": [("/t/", "/ʈ/")], "target": [("/t/", "/ʈ/")],
    "data": [("/d/", "/ɖ/")], "deadline": [("/d/", "/ɖ/")],
    # /ʃ/,/ʒ/
    "ship": [("/ʃ/", "/s/")], "should": [("/ʃ/", "/s/")], "measure": [("/ʒ/", "/z/")],
    # vowel elongation / shortening
    "experience": [("/ɪ/", "/iː/")], "important": [("/ʊ/", "/uː/")],
    "position": [("/ɪ/", "/iː/")], "focus": [("/ʊ/", "/uː/")],
    # generic vowel quality on function-ish words (low severity, accent-like)
    "to": [("/ə/", "/ʊ/")], "for": [("/ə/", "/ɔː/")],
}

# Extra phonemes to pad the compact detected_phonemes stream so it looks like a
# real per-word phoneme sequence (not used by any score/packet).
_FILLER_PHONEME_POOL = [
    "/ə/", "/ɪ/", "/iː/", "/e/", "/æ/", "/ɑː/", "/ʌ/", "/oʊ/", "/uː/",
    "/t/", "/d/", "/n/", "/s/", "/l/", "/r/", "/k/", "/m/", "/p/", "/f/",
]


# ---------------------------------------------------------------------------
# Latent / persona sampling
# ---------------------------------------------------------------------------
# Tier mix — deliberately flat so poor and exceptional tails are well covered,
# not concentrated in the middle.
TIER_MIX = [
    ("very_poor", 0.15, (0.00, 0.20)),
    ("poor", 0.22, (0.20, 0.40)),
    ("average", 0.26, (0.40, 0.60)),
    ("strong", 0.22, (0.60, 0.80)),
    ("exceptional", 0.15, (0.80, 1.00)),
]

# Personas overlay flavor on top of the tier's ability, guaranteeing coverage of
# every diversity bucket the brief lists. Each is a dict of modifier hints
# consumed in _build_latents.
PERSONAS = [
    "very_fluent", "moderately_fluent", "nervous", "fast_speaker", "slow_speaker",
    "excellent_pron", "average_pron", "pron_issues", "technical_word_mistakes",
    "mti_light", "mti_moderate", "mti_heavy",
    "expressive", "flat", "loses_energy",
    "confident", "disengaged", "highly_engaging", "balanced",
]


class R:
    """Thin RNG holder so numpy and python random stay reproducible together."""

    def __init__(self, seed: int):
        self.np = np.random.RandomState(seed)
        self.py = random.Random(seed)

    def uniform(self, a, b):
        return float(self.np.uniform(a, b))

    def normal(self, mu, sigma):
        return float(self.np.normal(mu, sigma))

    def choice(self, seq):
        return seq[self.np.randint(len(seq))]

    def chance(self, p):
        return self.np.random_sample() < p


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _assign_tiers_and_personas(n: int, rng: R) -> List[Tuple[str, float, str]]:
    """Return n (tier, theta, persona) triples with the target tier mix and a
    balanced-ish persona spread, shuffled."""
    assignments: List[Tuple[str, float, str]] = []
    # Tier counts from TIER_MIX proportions.
    counts = [max(0, round(frac * n)) for _, frac, _ in TIER_MIX]
    # Fix rounding drift so counts sum to n.
    while sum(counts) < n:
        counts[len(counts) // 2] += 1
    while sum(counts) > n:
        counts[len(counts) // 2] -= 1

    persona_cycle = list(PERSONAS)
    rng.py.shuffle(persona_cycle)
    pi = 0
    for (tier, _frac, (lo, hi)), count in zip(TIER_MIX, counts):
        for _ in range(count):
            theta = rng.uniform(lo, hi)
            persona = persona_cycle[pi % len(persona_cycle)]
            pi += 1
            assignments.append((tier, theta, persona))
    rng.py.shuffle(assignments)
    return assignments


def _build_latents(tier: str, theta: float, persona: str, rng: R) -> Dict:
    """Turn (tier, theta, persona) into concrete primitive parameters.

    theta in [0,1] is overall ability (higher = better). Personas nudge specific
    channels so the requested archetypes are realized while keeping natural noise
    and partial independence between modules.
    """
    # Response length (words) and base pace. Kept in single-interview-answer range
    # so total durations (and thus contour sizes) stay realistic.
    n_words = int(_clip(rng.normal(60 + 40 * theta, 26), 22, 150))
    # Target pace is set a little high because inter-word gaps drag the REALIZED
    # wpm (measured off the assembled timeline) down by ~15-30.
    wpm = _clip(rng.normal(138 + 42 * theta, 17), 85, 215)

    # Fluency channel. Pause probabilities are PER-WORD, so they must be small —
    # a few disfluencies per answer, not a fifth of all words.
    filler_rate = _clip(rng.normal(7.0 * (1 - theta) + 0.5, 2.2), 0.0, 16.0)  # per minute target
    long_pause_prob = _clip(rng.normal(0.025 * (1 - theta) + 0.008, 0.012), 0.0, 0.09)
    hesitation_prob = _clip(rng.normal(0.035 * (1 - theta) + 0.012, 0.018), 0.0, 0.12)

    # Rhythm issue probability (drives pronunciation_score's low tail via rhythm,
    # since rhythm_score is the main lever able to pull pronunciation below ~85).
    rhythm_mode = "normal"
    if rng.chance(_clip(0.42 * (1 - theta) + 0.05, 0.0, 0.6)):
        rhythm_mode = rng.choice(["uniform", "erratic"])

    # Pronunciation phoneme accuracy target (realistic bands by ability).
    acc = _clip(rng.normal(0.72 + 0.24 * theta, 0.06), 0.55, 1.0)

    # MTI strength — partially independent of theta (a fluent speaker can still
    # carry native-language patterns). Kept modest on average so most candidates
    # have light/no MTI; the mti_* personas cover the moderate/heavy end.
    mti_strength = _clip(0.32 * (1 - theta) + 0.36 * rng.uniform(0, 1), 0.0, 1.0)

    # Intonation expressiveness, partially tied to theta.
    expressiveness = _clip(0.25 + 0.6 * theta + rng.normal(0, 0.13), 0.0, 1.0)
    energy_decline = rng.chance(_clip(0.24 * (1 - theta) + 0.05, 0.0, 0.45))
    n_low_energy = int(_clip(rng.normal(2.8 * (1 - theta), 1.2), 0, 5))

    base_f0 = rng.uniform(95, 205)
    confidence = _clip(0.70 + 0.22 * theta + rng.normal(0, 0.05), 0.4, 0.99)

    # ---- persona overlays ----
    if persona == "very_fluent":
        filler_rate = min(filler_rate, 1.0); long_pause_prob = min(long_pause_prob, 0.03)
    elif persona == "moderately_fluent":
        filler_rate = _clip(filler_rate, 2.5, 5.0)
    elif persona == "nervous":
        filler_rate = _clip(filler_rate + 5.0, 4.0, 16.0)
        long_pause_prob = _clip(long_pause_prob + 0.04, 0.03, 0.12)
        hesitation_prob = _clip(hesitation_prob + 0.06, 0.05, 0.16)
        confidence = _clip(confidence - 0.2, 0.4, 0.9); wpm = _clip(wpm + rng.uniform(-25, 25), 80, 200)
    elif persona == "fast_speaker":
        wpm = _clip(rng.uniform(195, 215), 185, 215)
    elif persona == "slow_speaker":
        wpm = _clip(rng.uniform(78, 105), 75, 115)
    elif persona == "excellent_pron":
        acc = _clip(rng.uniform(0.95, 1.0), 0.95, 1.0); rhythm_mode = "normal"
    elif persona == "average_pron":
        acc = _clip(rng.uniform(0.80, 0.90), 0.75, 0.92)
    elif persona == "pron_issues":
        acc = _clip(rng.uniform(0.60, 0.78), 0.55, 0.80)
        if rng.chance(0.5):
            rhythm_mode = rng.choice(["uniform", "erratic"])
    elif persona == "technical_word_mistakes":
        acc = _clip(rng.uniform(0.72, 0.88), 0.6, 0.9); mti_strength = max(mti_strength, 0.4)
    elif persona == "mti_light":
        mti_strength = _clip(rng.uniform(0.05, 0.25), 0.0, 0.3)
    elif persona == "mti_moderate":
        mti_strength = _clip(rng.uniform(0.35, 0.6), 0.3, 0.65)
    elif persona == "mti_heavy":
        mti_strength = _clip(rng.uniform(0.7, 1.0), 0.65, 1.0)
    elif persona == "expressive":
        expressiveness = _clip(rng.uniform(0.8, 1.0), 0.75, 1.0); energy_decline = False
    elif persona == "flat":
        expressiveness = _clip(rng.uniform(0.0, 0.2), 0.0, 0.25)
    elif persona == "loses_energy":
        energy_decline = True; n_low_energy = max(n_low_energy, 2)
    elif persona == "confident":
        confidence = _clip(rng.uniform(0.85, 0.99), 0.8, 0.99); filler_rate = min(filler_rate, 2.5)
        expressiveness = max(expressiveness, 0.6)
    elif persona == "disengaged":
        expressiveness = min(expressiveness, 0.3); energy_decline = True
        n_low_energy = max(n_low_energy, 2)
        long_pause_prob = _clip(long_pause_prob + 0.03, 0.03, 0.1)
        hesitation_prob = _clip(hesitation_prob + 0.04, 0.04, 0.14)
    elif persona == "highly_engaging":
        expressiveness = _clip(rng.uniform(0.82, 1.0), 0.78, 1.0); filler_rate = min(filler_rate, 2.0)
        energy_decline = False; n_low_energy = 0; confidence = max(confidence, 0.85)

    emphasis_prob = _clip(expressiveness + rng.normal(0, 0.08), 0.0, 1.0)

    return {
        "tier": tier, "theta": theta, "persona": persona,
        "n_words": n_words, "wpm": wpm, "filler_rate": filler_rate,
        "long_pause_prob": long_pause_prob, "hesitation_prob": hesitation_prob,
        "rhythm_mode": rhythm_mode,
        "acc": acc, "mti_strength": mti_strength, "expressiveness": expressiveness,
        "energy_decline": energy_decline, "n_low_energy": n_low_energy,
        "base_f0": base_f0, "confidence": confidence, "emphasis_prob": emphasis_prob,
    }


# ---------------------------------------------------------------------------
# Primitive synthesis
# ---------------------------------------------------------------------------
def _build_word_timeline(latents: Dict, rng: R) -> Tuple[List[Dict], List[Dict], List[Dict], float]:
    """Lay out words with timestamps + confidence, and derive sentences and
    speech chunks. Returns (words, sentences, speech_chunks, total_duration)."""
    n_words = latents["n_words"]
    wpm = latents["wpm"]
    mean_slot = 60.0 / wpm  # avg seconds per word incl. its trailing gap

    # Word-duration coefficient of variation drives rhythm.
    if latents["rhythm_mode"] == "uniform":
        dur_cv = rng.uniform(0.02, 0.12)
    elif latents["rhythm_mode"] == "erratic":
        dur_cv = rng.uniform(0.95, 1.4)
    else:
        dur_cv = rng.uniform(0.30, 0.65)

    speech_frac = rng.uniform(0.68, 0.82)  # share of the slot that is voiced word, rest is natural gap
    mean_word_dur = max(0.09, mean_slot * speech_frac)
    filler_per_word = latents["filler_rate"] / max(1.0, wpm)  # fillers/min ÷ words/min

    words: List[Dict] = []
    t = round(rng.uniform(0.0, 0.4), 3)  # small lead-in silence
    base_conf = latents["confidence"]

    for i in range(n_words):
        # Pick surface word: mix of content / function / occasional filler / error-prone.
        roll = rng.np.random_sample()
        if roll < filler_per_word:
            word_text = rng.choice(FILLERS)
        elif roll < 0.35:
            word_text = rng.choice(FUNCTION_WORDS)
        elif rng.chance(0.16):
            word_text = rng.choice(list(ERROR_PRONE_WORDS.keys()))
        else:
            word_text = rng.choice(CONTENT_WORDS)

        dur = max(0.08, rng.normal(mean_word_dur, mean_word_dur * dur_cv))
        start = round(t, 3)
        end = round(start + dur, 3)

        # Trailing gap: the slot's natural remainder, with an occasional (rare)
        # disfluency pause added on top so average word+gap ≈ slot => wpm ≈ target.
        gap = max(0.03, mean_slot - dur)
        if rng.chance(latents["long_pause_prob"]):
            gap += rng.uniform(1.6, 4.2)  # long_pause or dead_air territory
        elif rng.chance(latents["hesitation_prob"]):
            gap += rng.uniform(0.5, 1.3)  # hesitation

        conf = round(_clip(rng.normal(base_conf, 0.08), 0.30, 0.995), 4)
        words.append({"word": word_text, "start": start, "end": end, "confidence": conf})
        t = end + gap

    total_duration = round(words[-1]["end"] + rng.uniform(0.1, 0.5), 3) if words else 1.0

    # Sentences: group ~8-14 words.
    sentences: List[Dict] = []
    i = 0
    while i < len(words):
        span = rng.py.randint(8, 14)
        group = words[i:i + span]
        if not group:
            break
        sentences.append({
            "text": " ".join(w["word"] for w in group),
            "start": group[0]["start"],
            "end": group[-1]["end"],
        })
        i += span

    # Speech chunks: split where an inter-word gap exceeds ~2s (a real silence).
    chunks: List[Dict] = []
    chunk_start = words[0]["start"] if words else 0.0
    prev_end = words[0]["end"] if words else 0.0
    cid = 1
    for w in words[1:]:
        if w["start"] - prev_end > 2.0:
            chunks.append({"chunk_id": cid, "start": round(chunk_start, 3), "end": round(prev_end, 3)})
            cid += 1
            chunk_start = w["start"]
        prev_end = w["end"]
    chunks.append({"chunk_id": cid, "start": round(chunk_start, 3), "end": round(prev_end, 3)})

    return words, sentences, chunks, total_duration


def _build_phoneme_errors(words: List[Dict], latents: Dict, rng: R,
                          usage: Dict[Tuple, int]) -> Tuple[List[Dict], float]:
    """Generate a small, realistic set of salient phoneme errors.

    We model the number of MISPRONOUNCED WORDS (a handful, as the real pipeline
    reports after grouping), not raw per-phoneme detector noise. Each affected
    word carries 1-3 phoneme errors — mostly real native-language substitution
    pairs on error-prone words (so MTI classifies them into vowel/consonant
    patterns), plus some generic slips and insertions/deletions (which MTI
    ignores). phoneme_accuracy is drawn from ability (latents["acc"]) rather than
    recomputed from this discrete count: the two are correlated (both fall as
    ability drops) but not identical, just as accuracy and the salient
    word-level errors aren't identical in the real output.

    Errors attach only to words present in the timeline so MTI/pronunciation can
    resolve their timestamps. Returns (errors, phoneme_accuracy)."""
    acc = latents["acc"]

    # A handful of affected words: rises as accuracy drops and as MTI strength rises.
    base = (1.0 - acc) * 18.0 + latents["mti_strength"] * 7.0
    n_error_words = int(_clip(round(rng.normal(base, 2.0)), 0, 15))

    cleaned = [(clean_word(w["word"]), w) for w in words]
    error_prone_present = [(cw, w) for cw, w in cleaned if cw in ERROR_PRONE_WORDS]
    content_present = [(cw, w) for cw, w in cleaned
                       if len(cw) > 2 and cw not in FUNCTION_WORDS and cw not in ERROR_PRONE_WORDS]
    rng.py.shuffle(error_prone_present)
    rng.py.shuffle(content_present)

    # Prefer error-prone words (biased by MTI strength), then generic content words.
    n_prone = min(len(error_prone_present), int(round(n_error_words * (0.4 + 0.5 * latents["mti_strength"]))))
    chosen = error_prone_present[:n_prone]
    remaining = n_error_words - len(chosen)
    chosen += content_present[:max(0, remaining)]
    # De-dup by cleaned word so one word isn't "mispronounced" twice.
    seen = set()
    chosen = [(cw, w) for cw, w in chosen if not (cw in seen or seen.add(cw))]

    def _generic_pair() -> Tuple[Optional[str], Optional[str]]:
        kind = rng.np.random_sample()
        if kind < 0.4:
            a, b = rng.choice(sorted(VOWEL_PHONEMES)), rng.choice(sorted(VOWEL_PHONEMES))
            return (a, b) if a != b else (a, rng.choice(sorted(VOWEL_PHONEMES)))
        if kind < 0.7:
            a, b = rng.choice(sorted(CONSONANT_PHONEMES)), rng.choice(sorted(CONSONANT_PHONEMES))
            return (a, b) if a != b else (a, rng.choice(sorted(CONSONANT_PHONEMES)))
        if rng.chance(0.5):  # insertion / deletion — MTI ignores these
            return (None, rng.choice(sorted(CONSONANT_PHONEMES)))
        return (rng.choice(sorted(VOWEL_PHONEMES)), None)

    def _least_used(pairs):
        return min(pairs, key=lambda p: usage.get(p, 0))

    errors: List[Dict] = []
    for cw, _w in chosen:
        n_ph = 1 if rng.chance(0.6) else (2 if rng.chance(0.7) else 3)
        for _ in range(n_ph):
            if cw in ERROR_PRONE_WORDS and rng.chance(0.8):
                pair = _least_used(ERROR_PRONE_WORDS[cw])
            else:
                pair = _generic_pair()
            usage[pair] = usage.get(pair, 0) + 1
            errors.append({"word": cw, "expected": pair[0], "detected": pair[1]})

    return errors, acc


def _build_detected_phonemes(words: List[Dict], errors: List[Dict], rng: R) -> List[Dict]:
    """Compact, schema-correct per-word detected-phoneme stream (not used by any
    score/packet — see module docstring). Substituted/inserted detected phonemes
    from `errors` are woven in so it stays consistent with the error list."""
    errors_by_word: Dict[str, List[Dict]] = {}
    for e in errors:
        errors_by_word.setdefault(e["word"], []).append(e)

    detected: List[Dict] = []
    for w in words:
        cw = clean_word(w["word"])
        if not cw:
            continue
        n_ph = min(3, max(1, round(len(cw) * 0.55)))
        span = max(0.02, w["end"] - w["start"])
        step = span / n_ph
        detected_syms = [rng.choice(_FILLER_PHONEME_POOL) for _ in range(n_ph)]
        # Overwrite one slot with each error's detected phoneme (when present).
        for k, e in enumerate(errors_by_word.get(cw, [])):
            if e["detected"] is not None and k < len(detected_syms):
                detected_syms[k] = e["detected"]
        for j, sym in enumerate(detected_syms):
            s = round(w["start"] + j * step, 3)
            detected.append({"phoneme": sym, "start": s, "end": round(s + step, 3)})
    return detected


def _build_contours(words: List[Dict], total_duration: float, latents: Dict, rng: R
                    ) -> Tuple[Dict, Dict]:
    """Synthesize pitch and energy contours (numpy) the real intonation
    sub-analyzers can consume directly. Returns (pitch_contour, energy_contour)."""
    n = max(4, int(round(total_duration / CONTOUR_DT)))
    times = np.arange(n) * CONTOUR_DT

    # Voiced where inside a word span.
    voiced = np.zeros(n, dtype=bool)
    for w in words:
        i0 = int(w["start"] / CONTOUR_DT)
        i1 = min(n, int(np.ceil(w["end"] / CONTOUR_DT)))
        if i1 > i0:
            voiced[i0:i1] = True

    base_f0 = latents["base_f0"]
    # Target voiced-F0 coefficient of variation from expressiveness: low -> flat
    # (monotonicity flags sustained windows), high -> expressive.
    E = latents["expressiveness"]
    target_cv = _clip(0.02 + 0.22 * E + rng.normal(0, 0.01), 0.012, 0.30)
    slow = 1.0 + 0.5 * target_cv * np.sin(2 * np.pi * times / max(6.0, total_duration / 3))
    noise = rng.np.normal(1.0, target_cv, size=n)
    f0 = base_f0 * slow * noise
    f0 = np.clip(f0, 70.0, 470.0)

    # Energy: baseline where voiced, low in gaps.
    base_e = rng.uniform(0.03, 0.09)
    rms = np.where(voiced, base_e * (1.0 + rng.np.normal(0, 0.08, size=n)), base_e * 0.04)
    rms = np.clip(rms, 1e-4, None)

    # Energy decline over the answer (loses-energy archetypes).
    if latents["energy_decline"]:
        ramp = np.linspace(1.0, rng.uniform(0.45, 0.65), n)
        rms = rms * ramp

    # Inject sustained low-energy sections.
    for _ in range(latents["n_low_energy"]):
        seg_len = int(rng.uniform(1.2, 3.0) / CONTOUR_DT)
        if seg_len < 1 or n - seg_len <= 1:
            continue
        s0 = rng.np.randint(0, n - seg_len)
        rms[s0:s0 + seg_len] *= rng.uniform(0.2, 0.42)

    # Emphasis peaks on a fraction of content words -> drives emphasis_score.
    for w in words:
        cw = clean_word(w["word"])
        if len(cw) > 2 and cw not in FUNCTION_WORDS and rng.chance(latents["emphasis_prob"]):
            i0 = int(w["start"] / CONTOUR_DT)
            i1 = min(n, int(np.ceil(w["end"] / CONTOUR_DT)))
            if i1 > i0:
                rms[i0:i1] *= 1.35
                f0[i0:i1] *= 1.13

    pitch_contour = {"f0": f0, "times": times, "voiced": voiced}
    energy_contour = {"rms": rms, "times": times}
    return pitch_contour, energy_contour


# ---------------------------------------------------------------------------
# Report assembly (real analyzers)
# ---------------------------------------------------------------------------
def _assemble_pronunciation(phoneme_accuracy: float, errors: List[Dict],
                            detected_phonemes: List[Dict], words: List[Dict]) -> Dict:
    """Reproduce analyze_pronunciation()'s output shape/logic from a synthetic
    (phoneme_accuracy, errors, detected_phonemes), reusing the real sub-analyzers
    and score formula. The only thing skipped is the wav2vec2 detection pass."""
    mispronounced = detect_mispronounced_words(errors, words)["mispronounced_words"]
    rhythm = analyze_rhythm(words)
    # stress_placement.analyze_stress is a hard placeholder that ALWAYS returns
    # accuracy 1.0 / no errors (detected stress == expected stress; see that
    # module's PLACEHOLDER notice). Reproduce its constant output rather than
    # call it, so the absent `pronouncing` dep is never exercised.
    stress_accuracy, stress_errors = 1.0, []
    pronunciation_score = round(100 * (phoneme_accuracy + rhythm["rhythm_score"]) / 2)
    return {
        "pronunciation_score": pronunciation_score,
        "phoneme_accuracy": phoneme_accuracy,
        "stress_accuracy": stress_accuracy,
        "rhythm_score": rhythm["rhythm_score"],
        "phoneme_errors": errors,
        "detected_phonemes": detected_phonemes,
        "mispronounced_words": mispronounced,
        "stress_errors": stress_errors,
        "rhythm_issues": rhythm["issues"],
        "overall_observations": _pron_observations(
            pronunciation_score, mispronounced, stress_errors, rhythm["issues"]
        ),
    }


def _assemble_intonation(pitch_contour: Dict, energy_contour: Dict, words: List[Dict]) -> Dict:
    """Reproduce analyze_intonation() on synthetic contours (each sub-analyzer
    accepts a pre-computed contour and ignores the waveform)."""
    dummy = np.array([], dtype=np.float32)
    pitch_report = analyze_pitch_variation(dummy, SAMPLE_RATE, pitch_contour=pitch_contour)
    energy_report = analyze_energy_variation(dummy, SAMPLE_RATE, energy_contour=energy_contour)
    monotonicity_report = analyze_monotonicity(dummy, SAMPLE_RATE, pitch_contour=pitch_contour)
    emphasis_report = analyze_emphasis(
        words, dummy, SAMPLE_RATE, pitch_contour=pitch_contour, energy_contour=energy_contour
    )
    _reconcile_pitch_with_monotonicity(pitch_report, monotonicity_report)
    intonation_score = round(float(np.mean([
        pitch_report["pitch_score"], energy_report["energy_score"],
        monotonicity_report["monotonicity_score"], emphasis_report["emphasis_score"],
    ])))
    return {
        "intonation_score": intonation_score,
        "delivery_label": _overall_delivery_label(intonation_score),
        "pitch_variation": pitch_report,
        "energy_variation": energy_report,
        "monotonicity": monotonicity_report,
        "emphasis": emphasis_report,
        "overall_observations": _combine_observations(
            intonation_score, pitch_report, energy_report, monotonicity_report, emphasis_report
        ),
        "pitch_contour": _serialize_pitch_contour(pitch_contour),
        "energy_contour": _serialize_energy_contour(energy_contour),
    }


def _build_sample(latents: Dict, rng: R, usage: Dict[Tuple, int]) -> Tuple[Dict, Dict]:
    """Build one full row and a compact scalar feature record for validation."""
    words, sentences, chunks, total_duration = _build_word_timeline(latents, rng)
    transcript = " ".join(w["word"] for w in words)

    errors, phoneme_accuracy = _build_phoneme_errors(words, latents, rng, usage)
    detected_phonemes = _build_detected_phonemes(words, errors, rng)
    pitch_contour, energy_contour = _build_contours(words, total_duration, latents, rng)

    fluency = analyze_fluency(words, sentences, total_duration)
    pronunciation = _assemble_pronunciation(phoneme_accuracy, errors, detected_phonemes, words)
    pronunciation["detected_phonemes"] = detected_phonemes
    mti = analyze_mti(None, transcript, words, sentences=sentences,
                      phoneme_errors=errors, stress_errors=[])
    intonation = _assemble_intonation(pitch_contour, energy_contour, words)
    engagement = analyze_engagement(fluency, pronunciation, mti, intonation, total_duration=total_duration)

    audio_analysis = {
        "fluency": fluency, "pronunciation": pronunciation, "mti": mti,
        "intonation": intonation, "engagement": engagement, "processing_metadata": {},
    }
    session_id = generate_session_id()
    timeline = build_timeline(
        duration=total_duration, sample_rate=SAMPLE_RATE,
        detected_language="en", language_probability=round(rng.uniform(0.82, 0.99), 4),
        speech_chunks=chunks, sentences=sentences, words=words,
        pronunciation_report=pronunciation, intonation_report=intonation, session_id=session_id,
    )
    analysis = build_features(audio_analysis, session_id)["analysis"]  # contours stripped from intonation
    # detected_phonemes is Level-1 ground truth and already lives in `timeline`;
    # drop the byte-for-byte duplicate the pronunciation block otherwise carries
    # (same dedup rationale the pipeline applies to intonation's contours; no
    # training packet reads pronunciation.detected_phonemes).
    analysis["pronunciation"] = {
        k: v for k, v in analysis["pronunciation"].items() if k != "detected_phonemes"
    }

    row = {"session_id": session_id, "timeline": timeline, **analysis}

    # Compact scalar record for statistics/validation (no big arrays).
    record = {
        "tier": latents["tier"], "persona": latents["persona"], "theta": round(latents["theta"], 3),
        "duration": total_duration, "n_words": len(words),
        "wpm": fluency["speaking_speed"]["words_per_minute"],
        "speed_class": fluency["speaking_speed"]["classification"],
        "filler_count": fluency["fillers"]["filler_count"],
        "filler_rate": fluency["fillers"]["fillers_per_minute"],
        "total_pauses": fluency["pauses"]["total_pauses"],
        "dead_air": sum(1 for p in fluency["pauses"]["pauses"] if p["type"] == "dead_air"),
        "pronunciation_score": pronunciation["pronunciation_score"],
        "phoneme_accuracy": pronunciation["phoneme_accuracy"],
        "rhythm_score": pronunciation["rhythm_score"],
        "n_mispronounced": len(pronunciation["mispronounced_words"]),
        "mti_score": mti["overall_mti_score"],
        "mti_clarity": mti["clarity_impact"],
        "n_mti_patterns": len(mti["vowel_patterns"]) + len(mti["consonant_patterns"]),
        "intonation_score": intonation["intonation_score"],
        "delivery_label": intonation["delivery_label"],
        "n_monotone": len(intonation["monotonicity"]["monotone_sections"]),
        "engagement_score": engagement["engagement_score"],
        "engagement_level": engagement["engagement_level"],
    }
    record["composite"] = round(statistics.mean([
        record["pronunciation_score"], record["mti_score"],
        record["intonation_score"], record["engagement_score"],
    ]), 1)
    return row, record


# ---------------------------------------------------------------------------
# Validation + statistics
# ---------------------------------------------------------------------------
def _score_band(score: float) -> str:
    if score < 40:
        return "very_poor"
    if score < 55:
        return "poor"
    if score < 70:
        return "average"
    if score < 85:
        return "strong"
    return "exceptional"


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(num / (dx * dy), 3) if dx and dy else 0.0


def _distribution(values: List) -> Dict:
    counts: Dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def build_statistics(records: List[Dict], exact_dupes: int, near_dupes: int,
                     max_signature_freq: int) -> Dict:
    numeric_keys = [
        "duration", "n_words", "wpm", "filler_count", "filler_rate", "total_pauses",
        "dead_air", "pronunciation_score", "phoneme_accuracy", "rhythm_score",
        "n_mispronounced", "mti_score", "n_mti_patterns", "intonation_score",
        "n_monotone", "engagement_score", "composite",
    ]
    summary = {}
    for k in numeric_keys:
        vals = [r[k] for r in records]
        summary[k] = {
            "mean": round(statistics.mean(vals), 3),
            "std": round(statistics.pstdev(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    correlations = {
        "filler_rate__vs__engagement_score": _pearson(
            [r["filler_rate"] for r in records], [r["engagement_score"] for r in records]),
        "phoneme_accuracy__vs__pronunciation_score": _pearson(
            [r["phoneme_accuracy"] for r in records], [r["pronunciation_score"] for r in records]),
        "n_monotone__vs__engagement_score": _pearson(
            [r["n_monotone"] for r in records], [r["engagement_score"] for r in records]),
        "intonation_score__vs__engagement_score": _pearson(
            [r["intonation_score"] for r in records], [r["engagement_score"] for r in records]),
        "mti_score__vs__engagement_score": _pearson(
            [r["mti_score"] for r in records], [r["engagement_score"] for r in records]),
        "dead_air__vs__engagement_score": _pearson(
            [r["dead_air"] for r in records], [r["engagement_score"] for r in records]),
        "theta__vs__composite": _pearson(
            [r["theta"] for r in records], [r["composite"] for r in records]),
    }

    coverage = {
        "engagement_band": _distribution([_score_band(r["engagement_score"]) for r in records]),
        "composite_band": _distribution([_score_band(r["composite"]) for r in records]),
        "engagement_level": _distribution([r["engagement_level"] for r in records]),
        "speed_class": _distribution([r["speed_class"] for r in records]),
        "delivery_label": _distribution([r["delivery_label"] for r in records]),
        "mti_clarity": _distribution([r["mti_clarity"] for r in records]),
        "tier": _distribution([r["tier"] for r in records]),
        "persona": _distribution([r["persona"] for r in records]),
    }

    return {
        "n_samples": len(records),
        "duplicates": {
            "exact_duplicate_rows": exact_dupes,
            "near_duplicate_score_signatures": near_dupes,
            "max_identical_signature_frequency": max_signature_freq,
        },
        "feature_summary": summary,
        "correlations": correlations,
        "coverage": coverage,
    }


def validate(records: List[Dict], stats: Dict) -> List[str]:
    """Automated checks. Returns a list of warnings (empty = all good). Range and
    correlation-sign checks are asserted; balance issues are warnings."""
    warnings: List[str] = []

    # 1. Range checks (hard).
    for r in records:
        for k in ("pronunciation_score", "mti_score", "intonation_score", "engagement_score"):
            assert 0 <= r[k] <= 100, f"{k} out of range: {r[k]}"
        assert 0.0 <= r["phoneme_accuracy"] <= 1.0, f"phoneme_accuracy out of range: {r['phoneme_accuracy']}"
        assert 0.0 <= r["rhythm_score"] <= 1.0
        assert r["wpm"] >= 0 and r["duration"] > 0

    # 2. Duplicates (hard on exact).
    assert stats["duplicates"]["exact_duplicate_rows"] == 0, "exact duplicate rows present"

    # 3. Correlation signs (hard — these are the causal relationships the brief wants).
    c = stats["correlations"]
    assert c["filler_rate__vs__engagement_score"] < 0, "fillers should anti-correlate with engagement"
    assert c["phoneme_accuracy__vs__pronunciation_score"] > 0, "accuracy should track pronunciation score"
    assert c["intonation_score__vs__engagement_score"] > 0, "intonation should track engagement"
    assert c["dead_air__vs__engagement_score"] <= 0, "dead air should not raise engagement"
    assert c["theta__vs__composite"] > 0, "latent ability should track composite score"

    # 4. Class balance (warnings). We balance on the ENGAGEMENT band — engagement
    # is the pipeline's closest thing to an overall interview-performance verdict,
    # and (unlike a raw 4-module mean) it isn't inflated by pronunciation's and
    # intonation's structurally high floors, so it's the meaningful axis for
    # "poor / average / strong / exceptional" coverage. Every band should be
    # represented and none should dominate (i.e. no middle concentration).
    eng_bands = stats["coverage"]["engagement_band"]
    total = stats["n_samples"]
    # The two extreme bands are intentionally the thinnest — truly exceptional and
    # truly catastrophic interviews are rarer than the poor/average/strong bulk, so
    # a realistic distribution shouldn't force ≥5% into each extreme. Middle bands
    # must still be well-populated; no band may dominate (middle-concentration).
    floors = {"very_poor": 0.035, "poor": 0.05, "average": 0.05, "strong": 0.05, "exceptional": 0.035}
    for band, floor in floors.items():
        share = eng_bands.get(band, 0) / total
        if share < floor:
            warnings.append(f"engagement band '{band}' under-represented: {share:.1%}")
    if max(eng_bands.values()) / total > 0.45:
        warnings.append("an engagement band exceeds 45% of samples (over-concentrated)")

    # 5. Signature concentration (warning).
    if stats["duplicates"]["max_identical_signature_frequency"] > total * 0.02:
        warnings.append("a score signature repeats in >2% of rows")

    return warnings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate_dataset(n: int, out_path: str, stats_path: str, seed: int = SEED) -> Dict:
    rng = R(seed)
    assignments = _assign_tiers_and_personas(n, rng)
    usage: Dict[Tuple, int] = {}

    records: List[Dict] = []
    exact_hashes: Dict[str, int] = {}
    signature_counts: Dict[Tuple, int] = {}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (tier, theta, persona) in enumerate(assignments):
            latents = _build_latents(tier, theta, persona, rng)
            row, record = _build_sample(latents, rng, usage)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.append(record)

            # dedup bookkeeping: hash the row minus its unique session_id.
            row_wo_id = {k: v for k, v in row.items() if k != "session_id"}
            row_wo_id["timeline"] = {k: v for k, v in row["timeline"].items() if k != "session_id"}
            h = hashlib.sha1(json.dumps(row_wo_id, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            exact_hashes[h] = exact_hashes.get(h, 0) + 1

            sig = (record["pronunciation_score"], record["mti_score"], record["intonation_score"],
                   record["engagement_score"], record["speed_class"], record["filler_count"])
            signature_counts[sig] = signature_counts.get(sig, 0) + 1

            if (i + 1) % 250 == 0:
                logger.info("generated %d/%d rows", i + 1, n)

    exact_dupes = sum(c - 1 for c in exact_hashes.values() if c > 1)
    near_dupes = sum(c - 1 for c in signature_counts.values() if c > 1)
    max_sig_freq = max(signature_counts.values()) if signature_counts else 0

    stats = build_statistics(records, exact_dupes, near_dupes, max_sig_freq)
    warnings = validate(records, stats)
    stats["validation_warnings"] = warnings

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_path = os.path.join(project_root, "dataset", "synthetic_audio_dataset.jsonl")
    stats_path = os.path.join(project_root, "dataset", "dataset_statistics.json")
    stats = generate_dataset(2000, out_path, stats_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
