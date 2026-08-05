# Synthetic Evidence Generation (Knowledge Distillation — Phase 0, rewrite)

Generates synthetic **raw deterministic evidence** — Level 1 (timeline) +
Level 2 (features) — that resembles production's own audio-pipeline output
closely enough to flow directly into `build_coach_packets()` unmodified.

**This module's only output is `packets/packets.jsonl`** (raw Level 1 +
Level 2 evidence). It does NOT call `build_coach_packets()` and does NOT
write Articulation/Delivery packets — that used to happen here and was
removed because it duplicated
`backend.knowledge_distillation.teacher_generation.packet_builder`, which
is now the sole place that derives coach packets from accepted evidence.
Run that module against this one's output to get Articulation/Delivery
packets.

## Model responsibilities (do not blur these)

```
Claude (this session)  ->  generates Level 1 + Level 2 evidence -- THIS module
MiniMax M3               ->  Articulation teacher outputs, a later stage
MiMo-v2.5                 ->  Delivery teacher outputs, a later stage
Gemma / Qwen               ->  student models, fine-tuned later on the
                                finished dataset -- NEVER used to generate it
```

**Qwen must not be introduced into this pipeline.** `provider.py` has no
concrete LLM-backed implementation — only an abstract interface — and
`config.py` never resolves `ZIQRA_TEACHER_*` (Qwen's credentials/model) for
any reason, even as a fallback default. This is enforced by a test
(`test_no_concrete_llm_backed_provider_exists_in_this_module`) that fails if
a concrete `Provider` subclass ever reappears in `provider.py`.

## Raw evidence only — no scores, no interpretation

**The dataset carries no scores, ratings, classifications, or
natural-language observations of any kind.** Only raw measurements, counts,
timestamps, and event data — exactly what a deterministic analyzer extracts
before any interpretation. No `pronunciation_score`, no `engagement_level`,
no `delivery_label`, no `overall_observations`, no
`strengths`/`improvement_areas`.

**One deliberate exception:** `severity` (low/medium/high) is kept on
`mispronounced_words[]` and MTI `vowel_patterns[]`/`consonant_patterns[]`
entries, even though it reads like a rating. `build_coach_packets()` reads
it via hard bracket indexing (`w["severity"]`), not `.get()` — every other
removed field degrades to `None` gracefully; these two crash the function
outright if absent. Confirmed by actually calling it, not assumed. No
other event anywhere carries a severity field.

```
Synthetic Generation Stage  ->  raw deterministic analytics only (THIS module)
                                 -> packets/packets.jsonl
Teacher Generation Stage    ->  build_coach_packets() derives Articulation/
                                 Delivery packets (backend.knowledge_
                                 distillation.teacher_generation)
Teacher Model Stage         ->  interpretation, reasoning, coaching, scoring
                                 (MiniMax M3 / MiMo-v2.5, over those packets
                                 -- not implemented yet)
```

`build_coach_packets()`'s own score/level output fields (e.g.
`articulation.pronunciation.score`) are legitimately `None` on every packet
derived from this module's evidence — that's correct, not a bug: this
stage never populates them, only the teacher model stage does later.

## How content is authored — in-session, no API call

Content is authored directly by Claude in this session (there's also no
mechanism for a script to invoke a Claude Code session as a callable API,
so this is the only path either way) and ingested via
`pipeline.ingest_authored()`, which runs the exact same
validate → diversity-check → accept/reject pipeline, just skipping the
prompt/provider round-trip:

```python
from backend.knowledge_distillation.synthetic_generation import (
    SpeakerBlueprint, SyntheticGenerationConfig, ingest_authored,
)

config = SyntheticGenerationConfig.from_env()
batch = [
    (SpeakerBlueprint(blueprint_id="bp_1", pronunciation="strong", ...), level1_dict, level2_dict),
    # ...
]
result = ingest_authored(batch, config, out_dir="backend/knowledge_distillation/synthetic_generation/datasets")
print(result)  # {"accepted": int, "rejected": int, "total_accepted_so_far": int, "rejections": [...]}
```

Safe to call repeatedly across many small batches: diversity state is
rebuilt from the existing `packets.jsonl` on every call (no separate state
file to go stale), and packets are appended, not overwritten.

`Pipeline.run()` (the automated, provider-driven loop) remains available
and fully tested for whenever a real `Provider` exists — construct one
explicitly (`Pipeline(blueprint_generator=..., evidence_generator=
EvidenceGenerator(your_provider, config), ...)`, see `__init__.py`'s
docstring). There is deliberately no `default_pipeline()` that
auto-constructs one: doing so today would mean silently wiring in whatever
`ZIQRA_TEACHER_*` points at (Qwen), which this phase must never call.

## Pipeline

```
SpeakerBlueprint             speaking behaviour only -- no role, seniority,
  (blueprint_generator)      or interview difficulty; no metrics, no scores
        |
        v
SyntheticEvidencePacket      Level 1 + Level 2, raw evidence only
  (evidence_generator /
   packet_from_authored)
        |
        v
ValidationResult             schema / range / logical / timeline
  (validator)                 consistency / production compatibility
        |
        v
DiversityResult               reject if overrepresented -- lightweight,
  (diversity_filter)           counter-based, no embeddings
        |
        v
Accepted dataset               reject-and-repeat, stops at
                                config.target_dataset_size ACCEPTED packets
```

## The blueprint

Eight dimensions (`config.BLUEPRINT_DIMENSIONS`), each a plain qualitative
band — no role, seniority, or question difficulty, because the deterministic
audio pipeline analyzes speech, not interview context:

| Dimension | Levels |
|---|---|
| `pronunciation` | weak / average / strong / excellent |
| `mti_severity` | none / light / moderate / heavy |
| `pace` | slow / moderate / fast |
| `filler_frequency` | minimal / moderate / frequent |
| `rhythm` | steady / uniform / erratic |
| `intonation` | flat / moderate / expressive |
| `engagement` | disengaged / neutral / engaging |
| `confidence` | low / moderate / high |

`energy` and `clarity` were folded in rather than made separate dimensions:
energy lives inside `intonation`, clarity inside `pronunciation`/`mti` —
matching production's own analyzer categories. The blueprint still shapes
authored content even though it's never scored: "weak pronunciation" means
the raw `phoneme_accuracy`/`mispronounced_words` should read that way, not
that a score field says so (there are no score fields).

`blueprint_generator.py` draws each dimension independently and uniformly at
random — no attempt to balance ahead of time. Balance is
`diversity_filter.py`'s job: reject an overrepresented draw and ask for
another. Tested at 200 and 2,000 accepted packets with zero diversity
rejections needed at either size — 8 independent uniform draws already
balance within the quota headroom by the time you have that many samples.

## Level 1 density

`acoustic_contours` / `detected_phonemes` are dense in real capture (~0.03s
hop, ~1 phoneme entry per 15 letters) but **`build_coach_packets()` never
reads Level 1 at all** (verified: it only reads the five Level 2 blocks). So
Level 1 is authored at a coarser, bounded density instead
(`config.CONTOUR_HOP_SECONDS = 0.25`, `MAX_DETECTED_PHONEMES_PER_WORD = 2`),
the same tradeoff `backend/distillation/synthetic_data_generator.py` already
made deliberately for the same reason.

## The Level 2 schema (raw fields only)

```
fluency:
  fillers: {filler_count, fillers_per_minute, fillers: [{word, start, end}]}
  pauses:  {total_pauses, pauses: [{start, end, duration, type}]}
  speaking_speed: {words_per_minute, sentences_per_minute}

pronunciation:
  phoneme_accuracy, stress_accuracy (always 1.0), rhythm_score
  phoneme_errors: [{word, expected, detected}]
  mispronounced_words: [{word, start, end}]
  stress_errors: [] (always empty), rhythm_issues: [] (always empty)

mti:
  summary: {vowel_pattern_issues, consonant_pattern_issues, stress_transfer_issues}
  vowel_patterns / consonant_patterns: [{word, timestamp, issue, expected, detected}]
  stress_transfer: [] (always empty)
  speech_statistics: {total_words_analyzed, affected_words, affected_percentage}

intonation:
  pitch_variation: {average_pitch, min_pitch, max_pitch, pitch_range}
  energy_variation: {low_energy_sections: [{start, end}]}
  monotonicity: {monotone_sections: [{start, end}]}
  emphasis: {under_emphasized_words: [{word, timestamp}]}

engagement: {}   # always empty -- entirely a derived synthesis in
                  # production, with no raw measurements of its own
```

Three field types deliberately survive even though they look categorical:
pause `type` (natural/hesitation/long_pause/dead_air/filled_pause) and MTI
`issue` (vowel_substitution/missing_aspiration/...) are **factual event
categories** — what kind of thing was detected — not a rating of how good or
bad it is. `severity` is the true exception described above (compatibility,
not a design preference) and is scoped to exactly two fields — nowhere else
carries it.

## Validation

Five independent checks, run in order (later ones skip if schema already
failed):

| Check | Catches |
|---|---|
| **schema** | missing required fields, wrong types, across both Level 1 and Level 2 |
| **range** | fractions outside [0,1], WPM/filler-rate outside sane bounds, `start >= end` |
| **logical** | raw numbers disagreeing with each other (see below) |
| **timeline_consistency** | Level 1 and Level 2 disagreeing with each other |
| **production_compatibility** | `build_coach_packets()` runs without raising (score/level fields being `None` in its output is expected, not a failure) |

**Logical validation** (all four checks compare raw numbers directly — no
label vs. metric, since no labels exist):
- high `phoneme_accuracy` (≥0.90) with more than 2 `mispronounced_words`
- `monotone_sections` covering ≥50% of the recording's duration with `pitch_range` still high (>120 Hz)
- `fillers[]` list length not matching `filler_count`
- `mti.summary`'s declared pattern counts not matching the actual pattern list lengths

**Timeline consistency** additionally checks: `words_per_minute` against the
actual Level 1 word count/duration, `fillers_per_minute` against
`filler_count`/duration, and that every word cited in `mispronounced_words`/
MTI patterns actually appears in `level1.words`.

**Production compatibility is checked by literally calling the real
function**, not a hand-maintained mirror of its field requirements — the
only way to actually guarantee "flows through without modification", and it
never goes stale if `coach_packets.py` changes. A malformed list item schema
validation can't catch (e.g. a `mispronounced_words` entry missing `"word"`)
still gets caught here, because `build_articulation_packet()` crashes on it
for real.

## Diversity

`diversity_filter.py` tracks two Counters: per-(dimension, level) accepted
counts, and exact 8-dimension-combination ("fingerprint") repeat counts. A
candidate blueprint is rejected if accepting it would push any single
dimension level's share past
`(1 / n_levels_for_that_dimension) × diversity_overrepresentation_factor`
(default 1.3×), or if its exact combination has already repeated
`max_exact_fingerprint_repeats` times (default 3). No embeddings, no
semantic similarity — deliberately lightweight.

## Provider interface / non-retryable failures

`provider.py` defines only `Provider` (a one-method ABC) — no concrete
implementation, no credentials, no `ZIQRA_TEACHER_*` reference anywhere (see
"Model responsibilities" above). Documented for a future concrete
implementation: transient failures should retry with exponential backoff;
account-level failures (`CreditsError`/auth errors) should not retry at all
— raise `NonRetryableProviderError` immediately. `Pipeline.run_iter()`
already treats that exception as a reason to abort the entire run (real,
tested code, exercised in `PipelineTests` with fake providers) — a future
concrete `Provider` gets this abort-the-whole-run behavior for free just by
raising the right exception type. `ingest_authored()`, the path actually
used today, never touches a provider so this doesn't come up there. It
exists because a *different* module's run once hit exactly this failure
mode against a different provider and spent 7 hours retrying an exhausted
account before being stopped manually — worth avoiding again whenever a
real provider is added here.

## Configuration

Every threshold, bound, and dimension lives in `config.py`; nothing is
hardcoded elsewhere.

```python
from backend.knowledge_distillation.synthetic_generation import SyntheticGenerationConfig

config = SyntheticGenerationConfig(target_dataset_size=200)
```

## Tests

```
venv/bin/python3 -m unittest backend.knowledge_distillation.synthetic_generation.tests.test_synthetic_generation -v
```

35 tests, all using a fake provider (no network, no credentials required).
`validator.py`'s production-compatibility check DOES call the real
`build_coach_packets()` — that's pure/local and needs no external service.
