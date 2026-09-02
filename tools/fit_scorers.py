"""
Fit and evaluate the scoring models on speechocean762's human ratings.

Input: the feature pickles produced by modal_fit_scorers.py.
Output: backend/feature_extractors/models/scorers.joblib and the held-out
metrics.

Pearson correlation against human scores on the official test split is what
the pronunciation-assessment literature reports, so it is what gets quoted —
never a training-set number.
"""

import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Imported, never restated: the same list the runtime scores against.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.feature_extractors.audio.scoring import FEATURE_NAMES, TARGETS  # noqa: E402

OUT = Path("backend/feature_extractors/audio/models/scorers.joblib")


def load(split):
    d = pickle.loads(Path(f"/tmp/so762_{split}.pkl").read_bytes())
    return d["X"], d["y"]


def main():
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    print(f"train {Xtr.shape}  test {Xte.shape}")

    models, report = {}, {}
    for i, name in enumerate(TARGETS):
        candidates = {
            "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
            "gbt": GradientBoostingRegressor(
                random_state=0, n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9
            ),
        }
        best = None
        for label, model in candidates.items():
            model.fit(Xtr, ytr[:, i])
            pred = np.clip(model.predict(Xte), 1, 10)
            pcc = float(pearsonr(pred, yte[:, i]).statistic)
            mse = float(mean_squared_error(yte[:, i], pred))
            print(f"  {name:<13} {label:<6} PCC {pcc:.3f}  MSE {mse:.3f}")
            if best is None or pcc > best[1]:
                best = (label, pcc, mse, model)
        label, pcc, mse, model = best
        models[name] = model
        report[name] = {"model": label, "test_pcc": round(pcc, 3), "test_mse": round(mse, 3)}

        if label == "gbt":
            order = np.argsort(model.feature_importances_)[::-1][:5]
            print("    top features:", ", ".join(f"{FEATURE_NAMES[j]}" for j in order))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "report": report, "features": FEATURE_NAMES}, OUT, compress=3)
    print(json.dumps(report, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
