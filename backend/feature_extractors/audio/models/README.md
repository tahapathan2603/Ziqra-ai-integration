# Fitted scorers

`scorers.joblib` holds the regressions that turn the pipeline's measurements
into the published pronunciation / fluency / delivery / overall scores.

They are fitted on **speechocean762** — 2,500 training utterances rated 1-10
by expert annotators, with 2,500 held out for test — and the bundle carries
its own held-out Pearson correlations under `report`, so any number quoted
about these scores can be checked against the file itself.

Refit:

```bash
python3 -m modal run modal_fit_scorers.py   # extracts features on a GPU
python3 tools/fit_scorers.py                # fits, evaluates, writes this file
```

The feature vector is defined once, in
`backend/feature_extractors/audio/scoring.py`, and imported by both the
trainer and the runtime — a second copy is how a model ends up scoring
different columns than it was fitted on.
