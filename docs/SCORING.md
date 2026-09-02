# How the scores are produced

Every published score is either a **measurement** (a count, a rate, a model
output) or a **regression fitted to human ratings**. Nothing is a
hand-weighted blend of hand-thresholded heuristics any more — that is what
this document exists to record, because the old design looked identical from
the outside and read as authoritative.

## What was wrong with the old scores

They did not discriminate, and the failures were measurable:

- Read-aloud Harvard sentences — about the least expressive speech there is —
  scored 87-96 "highly engaging".
- A clean 3.8s clip and a 24s accented one both scored pronunciation 96.
- A 4.5s answer reported intonation 96, engagement 96 and 83 words per minute.
  With no monotone stretch found, monotonicity returns 100; with no
  low-energy section, energy returns 100. The absence of a reading was
  published as a flattering one.
- `pronunciation_score` was 0.75 x phoneme match + 0.25 x a rhythm heuristic
  whose own docstring called it "directional, not authoritative".
- `engagement_score` was `0.30 energy + 0.25 pause + 0.35 dynamics +
  0.10 clarity` minus a coverage penalty. None of those weights had ever been
  compared against a human judgement.

## What produces them now

| Published score | Source |
| --- | --- |
| `pronunciation_score` | regression fitted to expert **accuracy** ratings |
| `fluency_score` | regression fitted to expert **fluency** ratings |
| `intonation_score` | regression fitted to expert **prosodic** ratings |
| `overall_score` | regression fitted to the raters' **total** |
| `words_per_minute`, `filler_count`, ... | direct measurements, unchanged |
| `vocal_arousal` | audeering MSP-Podcast model output, reported only |
| `audio_quality` | torchaudio SQUIM (STOI/PESQ), reported and used to gate |

The regressions are fitted on **speechocean762**: 2,500 utterances for
training and 2,500 held out, each rated 1-10 by expert annotators. Their
features are quantities the pipeline already computes — GOP over the wav2vec2
CTC log-probabilities, Praat pitch and energy statistics, rate — so scoring
adds five small regressions at request time, not another model.

Held-out Pearson correlations on the 2,500 test utterances the models never
saw:

| rated aspect | published as | PCC | MSE |
| --- | --- | --- | --- |
| fluency | `fluency_score` | 0.730 | 0.96 |
| prosodic | `intonation_score` | 0.724 | 0.98 |
| total | `overall_score` | 0.709 | 1.20 |
| accuracy | `pronunciation_score` | 0.684 | 1.28 |

Those are with the length-dependent features excluded (see
`scoring.LENGTH_DEPENDENT_FEATURES`). Keeping them scores better on the
benchmark — 0.724 / 0.786 / 0.777 / 0.748 — but `duration` came out as the top
feature for three of the four targets, which on a corpus of short read prompts
mostly means the model learned how long each prompt takes to read. There are
no prompts in a mock interview, and answers run three to six times longer than
anything in the training range, so the length-free models are the ones
shipped and the ~0.05 PCC is the price.

The numbers live inside `scorers.joblib` (`report`) as well, and
`tools/fit_scorers.py` prints them on every refit. Quote those, never a
training-set number.

What this changed on real recordings, fitted against the old heuristic:

| recording | pronunciation | delivery |
| --- | --- | --- |
| 24s accented read | 59 (was 96) | 55 (was 94) |
| 3.8s clean excerpt | 90 (was 96) | 88 (was 97) |
| 16s spontaneous | 72 (was 85) | 71 (was 89) |
| 23.8s at 5dB SNR | 69 (was 60) | 66 (was 93) |

The heuristics put everything between 85 and 97. These span 55 to 90.

## What is deliberately still a heuristic

`engagement_score`, flagged in the payload with
`engagement_score_is_heuristic: true`. speechocean762 rates pronunciation, not
how engaging someone is in an interview, so there is nothing here to fit it
against; renaming the raters' overall impression to "engagement" would be the
same unfounded leap the old design made. The app no longer shows it.

## What is measured but not scored

`vocal_arousal` — audeering's MSP-Podcast model, fitted to human arousal
ratings at CCC ~0.76-0.82. On this repo's recordings it ranked an energetic
sales pitch (28) below a mumbling teenager (34) and a monotone read passage
(38), so it is reported and not folded into any score. Arousal on naturalistic
podcast speech is not the same axis as engagement in a mock interview.

## When a score is withheld

`reliability` in the response lists anything suppressed and why:

- under ~8s of speech, the pitch family has too few windows to mean anything;
- under 15 words or 8s, a per-minute rate is extrapolation;
- a poor `audio_quality` verdict suppresses the pronunciation family, because
  noise degrades phoneme matching first and looks exactly like bad
  pronunciation (measured: the same recording at 5dB SNR scored 60 instead of
  96).
