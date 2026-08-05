# Teacher Generation (Knowledge Distillation)

Six pieces so far:

1. **`packet_builder.py`** — derives Articulation/Delivery **coach packets**
   from accepted synthetic evidence, via the real, unmodified
   `build_coach_packets()`. Calls no teacher model.
2. **`prompt_builder.py`** — turns a coach packet into the **prompt text**
   sent to that coach's teacher model. Calls no teacher model, builds no
   coach packets — imports neither `provider.py` nor `packet_builder.py`.
3. **`provider.py`** — the **Teacher Communication Layer**: sends a prompt
   to MiniMax M3 (Articulation) or MiMo-v2.5 (Delivery) and returns the raw
   response. Knows nothing about coach packets, prompts, or Level 1/2.
4. **`teacher_runner.py`** — the **Teacher Runner**: pure orchestration —
   `packet -> prompt_builder -> provider -> raw response`. No parsing, no
   validation, no saving, no retries (retries already live in `provider.py`;
   duplicating them here would just be two places doing the same job).
5. **`claude_teacher.py`** — `ClaudeTeacherProvider`, a temporary, local,
   drop-in replacement for `provider.py`'s `TeacherProvider` — same two
   methods, same signatures, zero network calls. See "Why claude_teacher.py
   exists" below.
6. **`generate_dataset.py`** — the pipeline-orchestration script: loops
   `TeacherRunner` over every session in the accepted evidence dataset and
   saves raw responses to disk. The one piece of this module that *is*
   allowed to loop the whole dataset and write files — everything above is
   deliberately per-session/stateless.

```
Level 1 Timeline
+
Level 2 Analytics
        |
        v
build_coach_packets()
        |
  ,-----+-----.
  v           v
Articulation   Delivery
   Packet       Packet
  |                |
  v                v
build_articulation_   build_delivery_
  prompt()             prompt()
  |                |
  v                v
prompt (str)     prompt (str)
  |                |
  v                v
.generate_articulation()  -> MiniMax M3
.generate_delivery()       -> MiMo-v2.5
  |
  v
raw response text (not parsed here -- a later stage's job)
```

`teacher_runner.py` ties the two together per-session (see "Usage — teacher
runner" below); `generate_dataset.py` loops that over the whole dataset and
saves results (see "Usage — generating the dataset" below). Response
validation and downstream dataset assembly are still **not implemented** —
later, not-yet-built stages.

## Why `claude_teacher.py` exists

A real-world run of `TeacherRunner()` (the default, real `TeacherProvider`)
against 3 sessions on 2026-08-04 surfaced two problems that made the full
2000-session x 2-coach run impractical as-is:

1. **Speed**: 30-90s per call (both models reason heavily before
   answering) — a strictly sequential run over all 4000 calls would take
   40-70+ hours.
2. **Reliability**: MiMo-v2.5 has a real empty-response failure mode —
   burns its token budget on internal reasoning and returns nothing, even
   after 3 retries, even at `max_tokens=4096`.

Per an explicit product decision, this iteration uses Claude itself (this
project's own synthetic-evidence-generation model, see
`synthetic_generation/README.md`) as a temporary stand-in teacher for
*both* coaches, instead of blocking the whole knowledge-distillation
pipeline on live MiniMax M3 / MiMo-v2.5 calls. `ClaudeTeacherProvider`
implements the exact evidence-grounded scoring/narration/reasoning a live
teacher would, as a deterministic, local generator (rules + templates)
instead of a network call.

**This is a drop-in, not a fork.** `ClaudeTeacherProvider` exposes the
identical two-method interface as `TeacherProvider`
(`.generate_articulation(prompt) -> str` / `.generate_delivery(prompt) ->
str`), so nothing in `prompt_builder.py`, `teacher_runner.py`, or
`generate_dataset.py` changed to support it — only which provider object
gets constructed. Swapping back to real teacher models later is a one-line
change (`TeacherRunner()` instead of
`TeacherRunner(provider=ClaudeTeacherProvider())`), once MiMo-v2.5's
reliability issue is addressed and/or the run is parallelized.

Every score is genuinely evidence-derived, not copied from any
pre-existing pipeline score (there are none in this dataset's evidence —
see "Score fields are `None`" below): `claude_teacher.py` implements its
own rule-based rubric per coach (phoneme-accuracy bands capped, not
subtracted, by mispronunciation severity; MTI pattern-count staircase;
wpm/filler/pause-driven fluency; pitch-range/flat-section-driven
intonation; engagement derived holistically from the other three, since
its own packet section carries no independent evidence in this dataset).
Verified against the real 2000-session dataset before the real run: zero
exceptions, and every rubric produces the full 1-5 range (an earlier
subtractive-penalty version of the scoring rules collapsed onto a single
floor score whenever low accuracy and multiple high-severity errors
co-occurred — a real, common combination in this data — silently skipping
score 2 entirely; fixed by capping instead of subtracting).

## Why packet derivation is a separate concern from communication

`backend.knowledge_distillation.synthetic_generation` used to build and
write Articulation/Delivery packets itself, inline, as part of accepting a
packet. That was removed: it duplicated exactly what `packet_builder.py`
does, and mixing "generate raw evidence" with "derive coach packets from
it" made the two concerns harder to test and reuse independently. Now:

- `synthetic_generation` produces **only** `packets/packets.jsonl` (raw
  Level 1 + Level 2 evidence).
- `teacher_generation.packet_builder` is the **sole** place that turns that
  evidence into coach packets.
- `teacher_generation.provider` is the **sole** place anything in
  `knowledge_distillation/` calls MiniMax M3 or MiMo-v2.5 — no other module
  imports an LLM client directly.

## Usage — packet derivation

```python
from backend.knowledge_distillation.teacher_generation import write_packet_pairs

count = write_packet_pairs(
    packets_path="backend/knowledge_distillation/synthetic_generation/datasets/packets/packets.jsonl",
    out_dir="backend/knowledge_distillation/teacher_generation/datasets",
)
```

Writes `datasets/articulation/articulation.jsonl` and
`datasets/delivery/delivery.jsonl`, one line each per session
(`{"session_id": ..., **build_coach_packets()'s value for that coach}`).

For streaming/one-at-a-time use instead of writing files:

```python
from backend.knowledge_distillation.teacher_generation import build_all

for pair in build_all(packets_path):
    pair.session_id, pair.articulation, pair.delivery  # CoachPacketPair
```

## Usage — prompt construction

```python
from backend.knowledge_distillation.teacher_generation import (
    build_articulation_prompt, build_delivery_prompt,
)

articulation_prompt = build_articulation_prompt(pair.articulation, pair.session_id)
delivery_prompt = build_delivery_prompt(pair.delivery, pair.session_id)
```

Each function takes the coach's packet dict (a `CoachPacketPair`'s
`.articulation` / `.delivery`) plus the session id, and returns a plain
prompt string — ready to pass straight to `TeacherProvider.
generate_articulation()` / `.generate_delivery()`. The prompt instructs the
teacher to produce a single JSON object with three top-level keys, in this
order: `scores` (1-5 per rubric this coach owns — Pronunciation + MTI for
Articulation, Fluency + Intonation + Engagement for Delivery), `coach_output`
(the team-locked schema from `docs/coach_output_schema.md`), and
`reasoning_trace` (evidence-cited explanation for every score and finding).
The coach packet is embedded verbatim as the prompt's only evidence source,
with explicit instructions not to invent observations beyond it.

## Usage — communication layer

```python
from backend.knowledge_distillation.teacher_generation import TeacherProvider

provider = TeacherProvider()
raw_text = provider.generate_articulation(prompt)  # -> MiniMax M3
raw_text = provider.generate_delivery(prompt)       # -> MiMo-v2.5
```

`TeacherProvider()` with no arguments never touches the network or the
environment at construction time — credentials/model/tuning are resolved
lazily from `ZIQRA_MINIMAX_*` (articulation) / `ZIQRA_MIM_*` (delivery) only
on the first real `generate_*` call, via `config.py`. Transient failures
(connection errors, empty responses) are retried with exponential backoff up
to `max_retries` times; account-level failures (no credits, bad API key)
raise `NonRetryableTeacherError` immediately, with no retry loop — this
distinction exists because an earlier version of this retry logic once spent
7 hours retrying a `CreditsError: Insufficient balance` before being caught
manually.

## Usage — teacher runner

```python
from backend.knowledge_distillation.teacher_generation import TeacherRunner, build_all

runner = TeacherRunner()
for pair in build_all(packets_path):
    raw_articulation = runner.run_articulation(pair.articulation, pair.session_id)  # -> MiniMax M3
    raw_delivery = runner.run_delivery(pair.delivery, pair.session_id)              # -> MiMo-v2.5
```

`run_articulation`/`run_delivery` each do exactly three things: build the
prompt via `prompt_builder`, send it via `provider`, return the raw response
text. Nothing is parsed, validated, retried (again — that's `provider.py`),
or saved; that's deliberately left to a later stage. `TeacherRunner()` with
no arguments builds a real `TeacherProvider()` internally, so it inherits
the same lazy credential resolution — constructing a `TeacherRunner` never
touches the network or environment either.

## Usage — generating the dataset

```
venv/bin/python3 -m backend.knowledge_distillation.teacher_generation.generate_dataset --provider claude
```

Loops `TeacherRunner` over every session in
`synthetic_generation/datasets/packets/packets.jsonl`, writing
`teacher_generation/datasets/raw_responses/articulation_raw.jsonl` and
`.../delivery_raw.jsonl` — one line per session,
`{"session_id", "generated_by", "raw_response"}`. `raw_response` is
whatever the provider returned, completely unparsed. `--provider real`
switches to live MiniMax M3 / MiMo-v2.5 calls; `--limit N` caps how many
sessions to process (useful for a smoke test before a full run).
Resumable: a rerun skips any `session_id` already present in the output
file, so an interrupted run costs nothing to restart. Aborts immediately
on `NonRetryableTeacherError` (account-level failure) rather than burning
through the rest of the dataset repeating the same failure; an ordinary
`TeacherProviderError` on one session is logged and skipped, left for the
next run to retry.

As of 2026-08-04: **2000 articulation + 2000 delivery records generated**,
all `"generated_by": "claude"`, all valid JSON matching the
`{scores, coach_output, reasoning_trace}` contract, zero errors.

## Score fields are `None` — expected, not a bug

`synthetic_generation`'s evidence carries no scores by design (see that
module's README). `build_coach_packets()` still runs correctly on it — its
own score/level fields (e.g. `articulation.pronunciation.score`) just come
out `None`. Scoring is the teacher model stage's job, not this one's.

## Errors

`build_packet_pair()` raises `PacketBuildError` if an evidence packet is
missing `session_id`/`level2`, or if `build_coach_packets()` itself raises
(e.g. a malformed `mispronounced_words` entry missing `"word"` — that
function reads it via hard indexing, not `.get()`). In practice this
shouldn't happen against `synthetic_generation`'s output, since its own
`validator.py` already confirms every accepted packet survives a real
`build_coach_packets()` call before acceptance — but `packet_builder.py`
doesn't assume that guarantee holds for evidence from anywhere else.

## Tests

```
venv/bin/python3 -m unittest \
    backend.knowledge_distillation.teacher_generation.tests.test_packet_builder \
    backend.knowledge_distillation.teacher_generation.tests.test_provider \
    backend.knowledge_distillation.teacher_generation.tests.test_prompt_builder \
    backend.knowledge_distillation.teacher_generation.tests.test_teacher_runner \
    backend.knowledge_distillation.teacher_generation.tests.test_claude_teacher \
    backend.knowledge_distillation.teacher_generation.tests.test_generate_dataset \
    -v
```

66 tests total, all local:

- `test_packet_builder.py` (9) — `build_coach_packets()` is pure/synchronous,
  no network call involved (enforced by
  `test_packet_builder_never_calls_a_teacher_model`).
- `test_provider.py` (9) — retry/backoff, non-retryable-error detection, and
  `generate_articulation`/`generate_delivery` routing, all against a fake
  `LLMClient` double (no network, no credentials). Its own module-boundary
  test, `test_provider_knows_nothing_about_coach_packets_or_prompts`, AST-
  scans `provider.py` to guarantee it never imports `packet_builder`,
  `teacher_generation.schemas`, `prompt_builder`, or `validator`.
- `test_prompt_builder.py` (15) — pure string assembly: correct teacher
  named, correct rubric set, output-contract key order (`scores` before
  `coach_output` before `reasoning_trace`), coach-output schema keys
  present, evidence embedded verbatim, no-hallucination/duplication-is-
  intentional instructions present. Its own module-boundary test AST-scans
  `prompt_builder.py` to guarantee it never imports `provider.py` or
  `packet_builder.py`.
- `test_teacher_runner.py` (8) — `run_articulation`/`run_delivery` build
  the exact `prompt_builder` output and hand it to the provider unchanged,
  return the provider's raw response untouched, and never cross-call the
  other coach's client, all against a fake `TeacherProvider` double (no
  network, no credentials). Its own module-boundary test AST-scans
  `teacher_runner.py` to guarantee it never imports `packet_builder`,
  `llm.client`/`llm.anthropic_client` (no duplicate API clients), or `json`
  (no parsing).
- `test_claude_teacher.py` (20) — prompt round-tripping (extracting the
  packet back out of a real `prompt_builder` prompt), scores always in
  range and evidence-grounded (clean evidence scores high, severe evidence
  scores low), drop-in compatibility with `TeacherRunner`. Its own
  module-boundary test AST-scans `claude_teacher.py` to guarantee it never
  imports an LLM SDK/client (`anthropic`, `openai`, `llm.client`,
  `llm.anthropic_client`) — the whole point of the module is that it
  doesn't.
- `test_generate_dataset.py` (5) — writes both output files tagged with
  the provider's `.name`, resume skips already-done sessions, `--limit`
  is respected, an ordinary `TeacherProviderError` is skipped (not fatal)
  while `NonRetryableTeacherError` aborts the whole run, all against a
  fake provider (no network, no credentials, tiny in-memory packets file).
