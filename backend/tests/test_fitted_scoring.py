"""
The fitted scorers' contract: feature order, graceful absence, and the
mapping onto the published 0-100 scale.

Feature order is load-bearing — it is the column order the regressions were
fitted on — so a change that reorders or renames a feature without refitting
must fail here rather than silently scoring against the wrong columns.
"""

import numpy as np

from backend.feature_extractors.audio import scoring


def _arrays(n=400, flat=False):
    rng = np.random.default_rng(0)
    f0 = np.full(n, 120.0) if flat else 120 + 30 * np.sin(np.linspace(0, 8, n))
    voiced = np.ones(n, dtype=bool)
    rms = 0.05 + 0.01 * rng.standard_normal(n)
    return f0, voiced, np.abs(rms)


def test_feature_vector_matches_the_declared_order():
    f0, voiced, rms = _arrays()
    vec = scoring.build_feature_vector(
        gop_scores=[-0.2, -0.5, -1.4, None, -3.0],
        f0=f0, voiced=voiced, rms=rms,
        expected_phonemes=40, duration_seconds=10.0, word_count=20,
        sample_rate=16000, hop_length=512,
    )
    assert vec is not None
    assert vec.shape == (len(scoring.FEATURE_NAMES),), "feature count must match the fitted columns"

    idx = {name: i for i, name in enumerate(scoring.FEATURE_NAMES)}
    # Spot-check the ones with arithmetic worth pinning.
    assert vec[idx["gop_count"]] == 4, "None entries are not scored phonemes"
    assert abs(vec[idx["gop_min"]] - (-3.0)) < 1e-9
    assert abs(vec[idx["gop_frac_below_2"]] - 0.25) < 1e-9
    assert abs(vec[idx["duration"]] - 10.0) < 1e-9
    assert abs(vec[idx["words_per_second"]] - 2.0) < 1e-9
    assert abs(vec[idx["phonemes_per_second"]] - 4.0) < 1e-9


def test_no_gop_means_no_feature_vector():
    """Without GOP there is nothing to score pronunciation from, and a vector
    of zeros would be scored confidently as terrible speech."""
    f0, voiced, rms = _arrays()
    assert scoring.build_feature_vector(
        gop_scores=[None, None], f0=f0, voiced=voiced, rms=rms,
        expected_phonemes=0, duration_seconds=5.0, word_count=5,
        sample_rate=16000, hop_length=512,
    ) is None


def test_flat_pitch_reads_as_flat_windows():
    n = 400
    flat_f0, voiced, rms = _arrays(n, flat=True)
    varied_f0, _, _ = _arrays(n, flat=False)
    flat = scoring.flat_window_fraction(flat_f0, voiced, 16000, 512)
    varied = scoring.flat_window_fraction(varied_f0, voiced, 16000, 512)
    assert flat > varied, "a constant pitch must register as flatter than a moving one"


def test_missing_bundle_degrades_instead_of_failing():
    """A missing model file must not take extraction down — the heuristic
    scores still stand."""
    out = scoring.score(None)
    assert out["fitted"] is False
    assert out["accuracy"] is None




def test_bundle_columns_are_selected_by_name():
    """The bundle records which features it was fitted on. Selecting them by
    name is what stops a feature added to FEATURE_NAMES later from shifting
    the columns underneath a model that never saw it."""
    assert set(scoring.SCORING_FEATURES) <= set(scoring.FEATURE_NAMES)
    assert set(scoring.LENGTH_DEPENDENT_FEATURES) & set(scoring.FEATURE_NAMES)
    assert not (set(scoring.SCORING_FEATURES) & set(scoring.LENGTH_DEPENDENT_FEATURES))

    bundle = scoring._get_bundle()
    if bundle is None:
        return  # bundle not present in this checkout; the loader test covers that
    assert bundle["columns"] == list(scoring.SCORING_FEATURES), (
        "the shipped bundle must agree with the module about its own columns"
    )
    for name, entry in bundle["report"].items():
        assert 0.0 <= entry["test_pcc"] <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
