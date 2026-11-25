"""Concordance for the second tier: the score that the constrained fine-tune maximises.

See docs/TIER2_CONCORDANCE_DESIGN.md. This file holds the smooth c-index; the practice windows and the
solvers will be added next to it.
"""
import numpy as np
import tensorflow as tf

DEFAULT_TAU = 0.1
FLAT_STD = 1e-6  # windows whose actuals vary less than this are not scored (as in SARIMAX)


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
