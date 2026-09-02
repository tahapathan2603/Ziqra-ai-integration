"""
Fit the pronunciation/fluency/prosody scorers on human ratings.

Replaces hand-picked thresholds and weights with regression fitted to
speechocean762: 2,500 train / 2,500 test utterances, each rated 1-10 by
expert annotators for accuracy, fluency, prosodic and completeness.

Features come from the pipeline's own components — GOP over the CTC
log-probs, Praat pitch/energy statistics, and rate — so the fitted model
scores using exactly what production already computes.

Reported on the held-out test split as Pearson correlation with the human
scores, which is what the pronunciation-assessment literature reports.

Run: python3 -m modal run modal_fit_scorers.py
"""

import io
import json
import pickle
from pathlib import Path

import modal

from modal_app import MODAL_REQUIREMENTS_FILE, _prefetch_models

# The production image ends with add_local_python_source, so nothing can be
# installed on top of it. This mirrors it and adds what only the fitting job
# needs: parquet reading for the dataset, and soundfile to decode its audio.
fit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install_from_requirements(MODAL_REQUIREMENTS_FILE)
    .pip_install("pyarrow==21.0.0", "soundfile==0.14.0")
    .run_function(_prefetch_models)
    .add_local_python_source("backend")
)

app = modal.App("ziqra-fit-scorers", image=fit_image)

PARQUET = {
    "train": "default/train/0000.parquet",
    "test": "default/test/0000.parquet",
}
REPO = "mispeech/speechocean762"

# Feature order lives in backend/.../scoring.py — the module inference uses —
# so the two cannot drift apart. Imported inside the remote function, since
# this file is also executed locally to define the app.


# Sharded rather than one long call. Two attempts at processing a whole
# 2,500-utterance split in one container were OOM-killed (exit 137) at
# roughly the same point, streaming or not, so something in the per-utterance
# path grows — parselmouth and phonemizer both hold C/C++ state across calls.
# A shard that exits every 500 utterances gives that memory back to the OS
# whatever the cause, and the shards run concurrently.
#
# retries=0: an OOM retry silently repeats the work and pays for it twice.
@app.function(gpu="A10G", timeout=3600, memory=16384, retries=0, max_containers=3)
def extract(split: str, start: int = 0, count: int = 500) -> bytes:
    """Feature matrix + labels for rows [start, start+count) of a split."""
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download

    from backend.feature_extractors.audio import scoring
    from backend.feature_extractors.audio.intonation.pitch_variation import (
        HOP_LENGTH, compute_pitch_contour,
    )
    from backend.feature_extractors.audio.pronunciation import gop
    from backend.feature_extractors.audio.pronunciation.phoneme_accuracy import (
        _get_model, get_expected_phonemes_for_sentence,
    )

    path = hf_hub_download(REPO, PARQUET[split], repo_type="dataset", revision="refs/convert/parquet")
    # Streamed in small batches rather than table.to_pylist(): the audio
    # column is ~330MB of parquet, which balloons into several gigabytes of
    # Python objects when materialised at once, and the container was
    # OOM-killed (exit 137) about a thousand utterances in — then retried
    # from the beginning, paying for the same work twice.
    parquet_file = pq.ParquetFile(path)
    total = min(count, max(0, parquet_file.metadata.num_rows - start))
    print(f"{split}[{start}:{start + total}]: {total} utterances", flush=True)

    processor, model, device = _get_model()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id

    features, labels, skipped = [], [], 0

    def stream():
        position, taken = 0, 0
        for batch in parquet_file.iter_batches(batch_size=16):
            rows = batch.to_pylist()
            for row in rows:
                if position >= start and taken < total:
                    yield row
                    taken += 1
                position += 1
            del rows, batch
            if taken >= total:
                return

    for i, row in enumerate(stream()):
        try:
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            text = row["text"]
            per_word = get_expected_phonemes_for_sentence(text)
            expected = [p for group in per_word if group for p in group]
            ids = [vocab.get(p) for p in expected]
            if not ids or any(v is None for v in ids):
                skipped += 1
                continue

            wav = torch.from_numpy(audio)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)[0].cpu()

            scores = gop.phoneme_gop(log_probs, ids, blank_id)
            del logits
            if not scores:
                skipped += 1
                continue
            g = np.array([s for s in scores if s is not None], dtype=np.float64)
            if g.size == 0:
                skipped += 1
                continue

            contour = compute_pitch_contour(audio, sr)
            import librosa as _lb

            rms = _lb.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]

            # The one definition of the feature vector, shared with inference
            # (backend/.../scoring.py). Restating it here is how a fitted model
            # ends up scoring different columns than it was trained on.
            vector = scoring.build_feature_vector(
                gop_scores=scores,
                f0=contour["f0"],
                voiced=contour["voiced"],
                rms=rms,
                expected_phonemes=len(expected),
                duration_seconds=len(audio) / sr,
                word_count=len(text.split()),
                sample_rate=sr,
                hop_length=HOP_LENGTH,
            )
            if vector is None:
                skipped += 1
                continue
            features.append(vector)
            labels.append([row["accuracy"], row["fluency"], row["prosodic"], row["completeness"], row["total"]])
            del log_probs, audio, row
        except Exception as err:
            skipped += 1
            if skipped < 5:
                print(f"  row {i} skipped: {type(err).__name__}: {err}", flush=True)

        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{total} done ({skipped} skipped)", flush=True)

    print(f"{split}[{start}]: {len(features)} usable, {skipped} skipped", flush=True)
    return pickle.dumps({"X": np.array(features), "y": np.array(labels, dtype=np.float64)})


@app.function(timeout=1800, memory=8192)
def fit(train_blob: bytes, test_blob: bytes) -> bytes:
    """
    Fits and evaluates in the container, and returns the joblib bytes.

    Deliberately not fitted locally: a GradientBoostingRegressor pickled by
    scikit-learn 1.7 fails to load under 1.9 ("No module named '_loss'"), so
    an artifact built on a laptop silently degrades the deployment to
    heuristic scores with one log line to say so. Fitting where it will be
    loaded removes the skew entirely.
    """
    import io
    import pickle as pkl

    import joblib
    import numpy as np
    from scipy.stats import pearsonr
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from backend.feature_extractors.audio.scoring import FEATURE_NAMES, SCORING_FEATURES, TARGETS

    train, test = pkl.loads(train_blob), pkl.loads(test_blob)
    # Fit on the length-invariant subset only — see scoring.SCORING_FEATURES.
    cols = [FEATURE_NAMES.index(name) for name in SCORING_FEATURES]
    Xtr, ytr = train["X"][:, cols], train["y"]
    Xte, yte = test["X"][:, cols], test["y"]
    print(f"train {Xtr.shape}  test {Xte.shape}  ({len(cols)} of {len(FEATURE_NAMES)} features)", flush=True)

    models, report = {}, {}
    for i, name in enumerate(TARGETS):
        best = None
        for label, model in (
            ("ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))),
            ("gbt", GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3,
                                              learning_rate=0.05, subsample=0.9)),
        ):
            model.fit(Xtr, ytr[:, i])
            pred = np.clip(model.predict(Xte), 1, 10)
            pcc = float(pearsonr(pred, yte[:, i]).statistic)
            mse = float(mean_squared_error(yte[:, i], pred))
            print(f"  {name:<13} {label:<6} PCC {pcc:.3f}  MSE {mse:.3f}", flush=True)
            if best is None or pcc > best[1]:
                best = (label, pcc, mse, model)
        label, pcc, mse, model = best
        models[name] = model
        report[name] = {"model": label, "test_pcc": round(pcc, 3), "test_mse": round(mse, 3)}

    print(json.dumps(report), flush=True)
    buf = io.BytesIO()
    joblib.dump(
        {"models": models, "report": report, "features": list(FEATURE_NAMES), "columns": list(SCORING_FEATURES)},
        buf,
        compress=3,
    )
    return buf.getvalue()


@app.local_entrypoint()
def main(shard: int = 500, rows: int = 2500):
    import numpy as np

    out = {}
    for split in ("train", "test"):
        jobs = [(split, start, shard) for start in range(0, rows, shard)]
        parts = [pickle.loads(blob) for blob in extract.starmap(jobs)]
        X = np.concatenate([p["X"] for p in parts if len(p["X"])])
        y = np.concatenate([p["y"] for p in parts if len(p["y"])])
        Path(f"/tmp/so762_{split}.pkl").write_bytes(pickle.dumps({"X": X, "y": y}))
        out[split] = int(X.shape[0])
        print(f"{split}: {X.shape}", flush=True)

    bundle = fit.remote(
        Path("/tmp/so762_train.pkl").read_bytes(), Path("/tmp/so762_test.pkl").read_bytes()
    )
    target = Path("backend/feature_extractors/audio/models/scorers.joblib")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bundle)
    print(f"wrote {target} ({len(bundle) / 1024:.0f} KB)")
    print(json.dumps(out))
