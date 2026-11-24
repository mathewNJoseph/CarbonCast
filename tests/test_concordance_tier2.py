"""Check the TF smooth c-index against the numpy version used in the SARIMAX study.

Run from the repository root with:  python -m pytest tests
"""
import numpy as np
import pytest
import tensorflow as tf
from scipy.special import expit

from concordance_tier2 import FLAT_STD, smooth_c


# Reference: copied unchanged from CarbonCast-SARIMAX-mat, src_clean/concordance_impl/concordance.py.
def smooth_concordance(actual, forecast, tau=0.1):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    i, j = np.triu_indices(len(a), k=1)
    da = a[j] - a[i]
    keep = da != 0
    if not keep.any():
        return float("nan")
    spread = f.std()
    if spread == 0:
        return 0.5
    oriented = (f[j] - f[i])[keep] * np.sign(da[keep])
    return float(np.mean(expit(oriented / (tau * spread))))


# Reference for many windows: SARIMAX's practice_concordance skips flat windows and averages the rest.
def mean_smooth_concordance(actuals, forecasts, tau=0.1):
    scores = [smooth_concordance(a, f, tau) for a, f in zip(actuals, forecasts) if np.std(a) >= FLAT_STD]
    return float(np.mean(scores)) if scores else float("nan")


@pytest.fixture
def windows():
    """Six 168 h windows, including the awkward cases: ties in the actuals, a flat window, a constant forecast."""
    rng = np.random.default_rng(0)
    actuals = rng.normal(size=(6, 168))
    forecasts = actuals + rng.normal(size=(6, 168))
    actuals[1] = np.round(actuals[1])  # many tied pairs
    actuals[2] = 0.4                   # flat actuals: skipped
    forecasts[3] = 0.0                 # constant forecast with std exactly 0: scores 0.5
    return actuals, forecasts


def test_matches_numpy_version(windows):
    actuals, forecasts = windows
    for tau in (0.1, 0.5):
        got = float(smooth_c(tf.constant(forecasts), actuals, tau))
        assert got == pytest.approx(mean_smooth_concordance(actuals, forecasts, tau), abs=1e-9)


def test_matches_numpy_version_per_window(windows):
    actuals, forecasts = windows
    for k in (0, 1, 3, 4, 5):
        got = float(smooth_c(tf.constant(forecasts[k:k + 1]), actuals[k:k + 1]))
        assert got == pytest.approx(smooth_concordance(actuals[k], forecasts[k]), abs=1e-9)


def test_float32_forecasts(windows):
    actuals, forecasts = windows
    got = smooth_c(tf.constant(forecasts, dtype=tf.float32), actuals)
    assert got.dtype == tf.float32
    assert float(got) == pytest.approx(mean_smooth_concordance(actuals, forecasts), abs=1e-5)


def test_stretching_the_forecast_does_not_change_the_score(windows):
    actuals, forecasts = windows
    assert float(smooth_c(tf.constant(7 * forecasts - 3), actuals)) == pytest.approx(
        float(smooth_c(tf.constant(forecasts), actuals)), abs=1e-12)


def test_all_flat_windows_give_nan():
    assert np.isnan(float(smooth_c(tf.constant(np.ones((2, 24))), np.full((2, 24), 3.0))))


def test_gradient_matches_finite_differences(windows):
    actuals, forecasts = windows
    x = tf.Variable(forecasts)
    with tf.GradientTape() as tape:
        c = smooth_c(x, actuals)
    grad = tape.gradient(c, x).numpy()
    assert np.isfinite(grad).all()  # including the constant-forecast window

    h = 1e-6
    for w, t in [(0, 0), (0, 100), (1, 50), (4, 167), (5, 20)]:
        up, down = forecasts.copy(), forecasts.copy()
        up[w, t] += h
        down[w, t] -= h
        numeric = (mean_smooth_concordance(actuals, up) - mean_smooth_concordance(actuals, down)) / (2 * h)
        assert grad[w, t] == pytest.approx(numeric, rel=1e-5, abs=1e-9)
    assert (grad[2] == 0).all()  # the flat window is not scored, so it has no gradient
