"""Tests for the output-layer concordance fine-tune in concordance_tier2 (variant A)."""
import numpy as np
import pytest

from concordance_tier2 import OutputLayerProblem, hard_c, practice_windows, slsqp_output_sweep


def brute_force_c(f, a):
    """Pair-by-pair c-index of one window: forecast ties count half, actual ties are skipped."""
    scores = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] != a[j]:
                scores.append(0.5 if f[i] == f[j] else float((f[j] - f[i]) * (a[j] - a[i]) > 0))
    return np.mean(scores)


def test_hard_c_matches_counting_pairs():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, size=(3, 30)).astype(float)   # ties in the actuals
    f = rng.integers(0, 5, size=(3, 30)).astype(float)   # and in the forecasts
    a[2] = 7.0                                           # a flat window is skipped
    assert hard_c(f, a) == pytest.approx(np.mean([brute_force_c(f[k], a[k]) for k in range(2)]))


def test_practice_windows_are_seven_consecutive_day_starts():
    w = practice_windows(4297)  # H1: 181 training days -> 180 day-ahead samples -> 174 windows
    assert w.shape == (174, 7)
    assert w[0].tolist() == [0, 24, 48, 72, 96, 120, 144] and w[-1, -1] == 24 * 179


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(24 * 20, 6))
    y = H @ rng.normal(size=(6, 24)) + 0.5 * rng.normal(size=(24 * 20, 24))
    return OutputLayerProblem(H, y, practice_windows(len(H), days=3))


def finite_difference(f, w, h=1e-6):
    return np.array([(f(w + h * e) - f(w - h * e)) / (2 * h) for e in np.eye(len(w))])


def test_gradients_match_finite_differences(problem):
    w = np.random.default_rng(2).normal(size=6 * 24 + 24)
    assert problem.mse_grad(w) == pytest.approx(finite_difference(problem.mse, w), rel=1e-5, abs=1e-8)
    assert problem.c_and_grad(w)[1] == pytest.approx(finite_difference(lambda v: problem.c_and_grad(v)[0], w),
                                                     rel=1e-4, abs=1e-7)


def test_sweep_stays_within_budget_and_gains_more_with_more_budget(problem):
    w0 = problem.least_squares() + 0.05  # a feasible start that is not the least-squares point
    results = slsqp_output_sweep(problem, w0, [0, 0.05, 0.2], max_iter=50, log=lambda m: None)
    c0 = problem.c_and_grad(w0)[0]
    cs = [results[e]["c"] for e in (0, 0.05, 0.2)]
    for eps, r in results.items():
        assert r["mse_ratio"] <= 1 + eps + 1e-9
    assert cs[0] >= c0 and cs[0] <= cs[1] <= cs[2] and cs[2] > c0


def small_model_problem():
    import tensorflow as tf
    from concordance_tier2 import ModelProblem
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24 * 12, 4, 3)).astype(np.float32)
    y = (X.sum(axis=(1, 2))[:, None] * np.linspace(-1, 1, 24) + rng.normal(size=(len(X), 24))).astype(np.float32)
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(4, 3)), tf.keras.layers.Dense(8, "relu"),
                                 tf.keras.layers.Dense(24)])
    model.compile("adam", "mse")
    model.fit(X, y, epochs=5, verbose=0)
    return model, ModelProblem(X, y, practice_windows(len(X), days=3))


@pytest.mark.parametrize("variables", ["output", "all"])
def test_switching_stays_within_budget_and_raises_concordance(variables):
    from concordance_tier2 import switching_sweep
    model, problem = small_model_problem()
    w0 = [w.copy() for w in model.get_weights()]
    c0, mse0 = float(problem.c(model)), float(problem.mse(model))
    results = switching_sweep(model, problem, [0.02, 0.2], variables, steps=60, check_every=5, lr_c=1e-2,
                              lr_mse=1e-2, batch_size=32, log=lambda m: None)
    assert all(np.array_equal(a, b) for a, b in zip(model.get_weights(), w0))  # the sweep restores the start
    for eps, r in results.items():
        model.set_weights(r["weights"])
        assert float(problem.mse(model)) / mse0 == pytest.approx(r["mse_ratio"], rel=1e-5)
        assert r["mse_ratio"] <= 1 + eps and r["c"] >= c0
        if variables == "output":  # only the final Dense layer moved
            assert all(np.array_equal(a, b) for a, b in zip(r["weights"][:-2], w0[:-2]))
    assert results[0.2]["c"] > c0


def test_forward_in_chunks_matches_predict():
    from concordance_tier2 import forward
    model, problem = small_model_problem()
    assert forward(model, problem.X, chunk=50).numpy() == pytest.approx(model.predict(problem.X, verbose=0), abs=1e-6)
