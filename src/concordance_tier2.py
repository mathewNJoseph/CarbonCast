"""Concordance for the second tier: the smooth c-index and the constrained concordance fine-tune.

See docs/TIER2_CONCORDANCE_DESIGN.md. The problem, for each eps in a grid, is

    maximise    C_A(theta)                      smooth c-index over the practice windows of the training data
    subject to  MSE(theta) <= (1 + eps) * MSE(theta_0)

Three solvers (section 5.3), all on the same objective and constraint:
- slsqp_output: SLSQP on the 600 weights of the final Dense(24) layer, everything below frozen (section 5.1).
- switching_output: the switching gradient on the same 600 weights (section 5.2), to check it against SLSQP.
- switching_all: the switching gradient on every weight.
C and MSE are never summed or blended: each step or iterate serves one of them, and the answer for each eps is
the most concordant point that satisfies the constraint.
"""
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

DEFAULT_TAU = 0.1
FLAT_STD = 1e-6  # windows whose actuals vary less than this are not scored (as in SARIMAX)
FEASIBLE_TOL = 1e-9  # relative slack allowed on the MSE bound when picking iterates


def smooth_c(forecasts, actuals, tau=DEFAULT_TAU):
    """Mean smooth c-index over windows, differentiable with respect to the forecasts.

    A TF port of smooth_concordance in CarbonCast-SARIMAX-mat (src_clean/concordance_impl/concordance.py),
    batched over windows. For each window, every pair of hours whose actuals differ scores
    sigmoid(oriented forecast difference / (tau * std of the forecast)); the window's score is the mean over
    those pairs. Dividing by the forecast's std means stretching the forecast cannot raise the score.
    Windows with flat actuals are skipped, and the result is the mean over the rest.

    forecasts: tensor of shape (windows, hours). Gradients flow through it.
    actuals:   array of the same shape. Treated as constants.
    Returns a scalar tensor with the forecasts' dtype, or NaN if every window is flat.
    """
    forecasts = tf.convert_to_tensor(forecasts)
    actuals = np.asarray(actuals, dtype=np.float64)
    scored = actuals.std(axis=1) >= FLAT_STD
    if not scored.any():
        return tf.constant(np.nan, dtype=forecasts.dtype)
    forecasts = tf.boolean_mask(forecasts, scored)
    actuals = actuals[scored]

    i, j = np.triu_indices(actuals.shape[1], k=1)
    sign = np.sign(actuals[:, j] - actuals[:, i])  # +1 or -1, or 0 for tied actuals
    untied = sign != 0

    # A constant forecast has std 0. Divide by 1 instead: every difference is 0, so each pair scores
    # sigmoid(0) = 0.5, as in the numpy version.
    variance = tf.math.reduce_variance(forecasts, axis=1, keepdims=True)  # population variance, as np.std
    spread = tf.sqrt(tf.where(variance > 0, variance, tf.ones_like(variance)))

    diff = tf.gather(forecasts, j, axis=1) - tf.gather(forecasts, i, axis=1)
    pair_scores = tf.sigmoid(tf.cast(sign, forecasts.dtype) * diff / (tau * spread))
    untied = tf.cast(untied, forecasts.dtype)
    per_window = tf.reduce_sum(pair_scores * untied, axis=1) / tf.reduce_sum(untied, axis=1)
    return tf.reduce_mean(per_window)


def hard_c(forecasts, actuals):
    """Mean c-index over windows: the share of pairs with different actuals that the forecast orders the same
    way, counting a forecast tie as half. Windows with flat actuals are skipped. This is lifelines'
    concordance_index without censoring, which SARIMAX uses to score its forecast weeks.
    """
    forecasts, actuals = np.asarray(forecasts, dtype=np.float64), np.asarray(actuals, dtype=np.float64)
    scored = actuals.std(axis=1) >= FLAT_STD
    forecasts, actuals = forecasts[scored], actuals[scored]
    i, j = np.triu_indices(actuals.shape[1], k=1)
    sign_a = np.sign(actuals[:, j] - actuals[:, i])
    sign_f = np.sign(forecasts[:, j] - forecasts[:, i])
    untied = sign_a != 0
    score = np.where(sign_f == 0, 0.5, (sign_f == sign_a).astype(np.float64))
    return float(np.mean((score * untied).sum(axis=1) / untied.sum(axis=1)))


def practice_windows(num_samples, days=7):
    """Sample indices of C_A's practice windows, shape (windows, days).

    Sample 24*d of manipulateTrainingDataShape is "history = day d, predict day d+1". Window w is the
    teacher-forced predictions for days w+1 .. w+days, the same 168 hours a test forecast covers.
    """
    day = np.arange(0, num_samples, 24)
    count = len(day) - days + 1
    return day[np.arange(count)[:, None] + np.arange(days)]


def penultimate_features(model, X):
    """Output of the layer below the final Dense, dropout off, as float64 of shape (samples, units)."""
    if not isinstance(model.layers[-1], tf.keras.layers.Dense):
        raise ValueError(f"expected the model to end in Dense, got {type(model.layers[-1]).__name__}")
    features = tf.keras.Model(model.inputs, model.layers[-2].output)
    return features.predict(X, batch_size=1024, verbose=0).astype(np.float64)


class OutputLayerProblem:
    """C_A and the training MSE as functions of the final Dense layer's weights, packed as one vector."""

    def __init__(self, H, y, windows, tau=DEFAULT_TAU):
        self.H, self.y, self.tau = H, y.astype(np.float64), tau
        self.units, self.outputs = H.shape[1], y.shape[1]
        self.window_H = tf.constant(H[windows])                         # (windows, days, units)
        self.window_actuals = self.y[windows].reshape(len(windows), -1)  # (windows, days * outputs)
        self._cached = (None, None)

    def pack(self, kernel, bias):
        return np.concatenate([np.ravel(kernel), bias]).astype(np.float64)

    def unpack(self, w):
        split = self.units * self.outputs
        return w[:split].reshape(self.units, self.outputs), w[split:]

    def mse(self, w):
        kernel, bias = self.unpack(w)
        return float(np.mean((self.H @ kernel + bias - self.y) ** 2))

    def mse_grad(self, w):
        kernel, bias = self.unpack(w)
        residual = (self.H @ kernel + bias - self.y) * (2.0 / self.y.size)
        return self.pack(self.H.T @ residual, residual.sum(axis=0))

    def c_and_grad(self, w):
        """Smooth C_A and its gradient. SLSQP asks for the value and the gradient at the same point
        separately, so the last result is cached."""
        if self._cached[0] is not None and np.array_equal(self._cached[0], w):
            return self._cached[1]
        kernel, bias = (tf.Variable(a) for a in self.unpack(w))
        with tf.GradientTape() as tape:
            f = tf.einsum("wdu,uo->wdo", self.window_H, kernel) + bias
            c = smooth_c(tf.reshape(f, (self.window_actuals.shape[0], -1)), self.window_actuals, self.tau)
        dk, db = tape.gradient(c, [kernel, bias])
        result = (float(c), self.pack(dk.numpy(), db.numpy()))
        self._cached = (w.copy(), result)
        return result

    def least_squares(self):
        """The weights that minimise the training MSE outright, for the MSE_LS / MSE0 slack check."""
        design = np.hstack([self.H, np.ones((len(self.H), 1))])
        solution = np.linalg.lstsq(design, self.y, rcond=None)[0]
        return self.pack(solution[:-1], solution[-1])


def best_within(points, problem, bound):
    """The most concordant point whose MSE is within bound, as (w, C, MSE), or None."""
    best = None
    for w in points:
        m = problem.mse(w)
        if m <= bound * (1 + FEASIBLE_TOL):
            c = problem.c_and_grad(w)[0]
            if best is None or c > best[1]:
                best = (w, c, m)
    return best


def slsqp_output_sweep(problem, w0, eps_grid, max_iter=200, log=print):
    """Variant A: for each eps, in increasing order, run SLSQP from w0 and from the previous eps's answer, pool
    every iterate and keep the most concordant feasible one (w0 itself is always feasible).
    Returns {eps: {"w", "c", "mse_ratio", "seconds"}}."""
    mse0 = problem.mse(w0)
    results, previous = {}, w0
    for eps in sorted(eps_grid):
        start_time = time.time()
        points = [w0, previous]
        for start in ([w0] if previous is w0 else [w0, previous]):
            res = minimize(lambda w: -problem.c_and_grad(w)[0], start, jac=lambda w: -problem.c_and_grad(w)[1],
                           method="SLSQP",
                           # Scaled by MSE0 so the constraint is about 1 in size, like the objective.
                           constraints=[{"type": "ineq",
                                         "fun": lambda w: 1 + eps - problem.mse(w) / mse0,
                                         "jac": lambda w: -problem.mse_grad(w) / mse0}],
                           callback=lambda wk: points.append(wk.copy()),
                           options={"maxiter": max_iter})
            points.append(res.x)
        w, c, m = best_within(points, problem, (1 + eps) * mse0)
        results[eps] = {"w": w, "c": c, "mse_ratio": m / mse0, "seconds": time.time() - start_time}
        log(f"  eps {eps:g}: train C_A {c:.4f}, MSE ratio {m / mse0:.4f} "
            f"({len(points)} points, {results[eps]['seconds']:.1f}s)")
        previous = w
    return results


class ModelProblem:
    """C_A and the training MSE of a whole Keras model, for the switching gradient. Dropout is off throughout,
    so the optimizer targets exactly what is evaluated. The model is passed to each call, so one problem can
    score the baseline and any variant."""

    def __init__(self, X, y, windows, tau=DEFAULT_TAU):
        self.X, self.y, self.tau = tf.constant(X, tf.float32), tf.constant(y, tf.float32), tau
        self.window_X = tf.constant(X[windows.ravel()], tf.float32)
        self.window_actuals = y[windows].reshape(len(windows), -1)

    def c(self, model):
        f = forward(model, self.window_X)
        return smooth_c(tf.reshape(f, (self.window_actuals.shape[0], -1)), self.window_actuals, self.tau)

    def mse(self, model, rows=None):
        X, y = (self.X, self.y) if rows is None else (tf.gather(self.X, rows), tf.gather(self.y, rows))
        return tf.reduce_mean(tf.square(forward(model, X) - y))


def forward(model, X, chunk=512):
    """model(X) with dropout off, in chunks. On the Apple Metal GPU, one call over a few thousand samples returns
    wrong outputs (off by 0.1 in scaled units for the AECI tier 2 model) while chunks match model.predict."""
    return tf.concat([model(X[i:i + chunk], training=False) for i in range(0, X.shape[0], chunk)], axis=0)


def switching_sweep(model, problem, eps_grid, variables, steps=300, check_every=10, lr_c=1e-3, lr_mse=1e-3,
                    batch_size=256, seed=0, log=print):
    """Variants B and C: Polyak's switching gradient (design section 5.2), on the output layer
    (variables="output") or every weight (variables="all").

    For each eps the model starts from the baseline's weights. Every check_every steps the full-data MSE ratio
    and C_A are checked; the run then climbs C_A (full batch) while the ratio is within 1+eps and descends the
    MSE (minibatches) while it is not, with a separate Adam optimizer for each so their moments do not mix. Each
    step follows one objective only; the answer is the most concordant checked point that satisfied the bound.
    model is modified in place. Returns {eps: {"weights", "c", "mse_ratio", "seconds"}}.
    """
    w0 = model.get_weights()
    trainable = {"output": model.layers[-1].trainable_variables, "all": model.trainable_variables}[variables]
    mse0 = float(problem.mse(model))
    rng = np.random.default_rng(seed)
    results = {}
    for eps in sorted(eps_grid):
        start_time = time.time()
        model.set_weights(w0)
        opt_c, opt_mse = tf.keras.optimizers.Adam(lr_c), tf.keras.optimizers.Adam(lr_mse)
        best, mode, feasible_checks = None, "C", 0
        for step in range(steps + 1):
            if step % check_every == 0 or step == steps:
                ratio, c = float(problem.mse(model)) / mse0, float(problem.c(model))
                feasible = ratio <= 1 + eps
                if feasible:
                    feasible_checks += 1
                    if best is None or c > best[1]:
                        best = (model.get_weights(), c, ratio)
                mode = "C" if feasible else "MSE"
            if step == steps:
                break
            with tf.GradientTape() as tape:
                if mode == "C":
                    loss, opt = -problem.c(model), opt_c
                else:
                    rows = rng.choice(problem.X.shape[0], size=batch_size, replace=False)
                    loss, opt = problem.mse(model, rows), opt_mse
            opt.apply_gradients(zip(tape.gradient(loss, trainable), trainable))
        weights, c, ratio = best  # the start is the baseline, which is always feasible
        results[eps] = {"weights": weights, "c": c, "mse_ratio": ratio, "seconds": time.time() - start_time}
        log(f"  eps {eps:g}: train C_A {c:.4f}, MSE ratio {ratio:.4f} "
            f"({feasible_checks} of {steps // check_every + 1} checks feasible, {results[eps]['seconds']:.1f}s)")
    model.set_weights(w0)
    return results
