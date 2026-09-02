# ziqra_ai_service — speech analysis

One recording in, JSON out: Level 1 (transcript, word and sentence
timings, pitch/energy contours, detected phonemes) and Level 2 (fluency,
pronunciation, MTI, intonation, engagement) features and scores. The API is
`backend/api/main.py` (FastAPI: `GET /health`, `POST /extract` behind an
`X-Client-Token` header); `modal_app.py` is deployment only — it puts that
same app object on one A10G Modal container and serves it over HTTP, so a
local `uvicorn backend.api.main:app` runs exactly what production runs.

## The published scores are fitted to human ratings

Read **[`docs/SCORING.md`](docs/SCORING.md)** before touching or quoting any
score. The headline numbers — `pronunciation_score`, `fluency_score`,
`intonation_score`, `overall_score` — are no longer hand-weighted blends of
hand-thresholded heuristics; they are regressions fitted to speechocean762's
expert 1-10 ratings over features the pipeline already computes (GOP from the
wav2vec2 CTC log-probabilities, Praat pitch and energy statistics, rate), so
no extra model runs at request time. Held out on the 2,500 test utterances the
models never saw, Pearson correlation is 0.730 for fluency, 0.724 for prosodic
(`intonation_score`), 0.709 for the raters' total (`overall_score`) and 0.684
for accuracy (`pronunciation_score`); the old composites, by contrast, put
every real recording between 85 and 97 and scored read-aloud Harvard sentences
87-96 "highly engaging". `engagement_score` is deliberately still a heuristic
and flagged as one in the payload (`engagement_score_is_heuristic: true`),
because speechocean762 rates pronunciation, not how engaging someone is —
`docs/SCORING.md` has that reasoning, the full feature/score table, and why a
metric is sometimes withheld instead of guessed.

## A cold container costs ~28s; warm requests cost ~1.4s

Measured on an A10G, three identical requests with a 4.5s clip through one
container: 28.5s, then 1.37s, then 1.36s. Practically all of that first number
is one-time per container — librosa/numba JIT compilation on the first
pitch/energy call, plus loading Whisper and the wav2vec2 phoneme model — so
the pipeline was never slow, it was cold on arrival.
`backend/api/warmup.py` pays that cost at container start instead, in a
background thread kicked off by the ASGI app's startup event: it loads
Whisper, loads the phoneme model and runs one tiny inference through it so its
CUDA kernels compile too, and pushes two seconds of synthetic voiced signal
through the pitch, energy, monotonicity, emphasis, arousal and SQUIM paths so
their code is compiled and their weights loaded before anyone speaks. A warm-up failure is logged and swallowed, never fatal —
the first real request just pays the cost as it used to.

`GET /health` therefore reports readiness as well as liveness: `warm` and
`warmup_seconds` alongside `status`. It never blocks by default, so it stays a
cheap probe, but **`/health?wait=<seconds>`** blocks up to that long (capped at
120s) for warm-up to finish, which is what a caller who wants a
guaranteed-warm container should use. The app pre-warms at the start of
onboarding precisely to absorb this; before warm-up existed, `/health` touched
no model, so that ping only booted a container and left the whole 28s for the
user's first spoken answer. End to end through the public endpoint, HTTP
included, a 4.5s answer now takes 1.47s warm and a 26s answer 3.33s.
`scaledown_window` is 900s rather than 300s for the same reason: an interview
is answers separated by thinking time, and five minutes was short enough to
drop the container mid-session and charge one candidate that cost twice.
