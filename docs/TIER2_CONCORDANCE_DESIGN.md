# Tier 2 concordance: detailed design

Status: **design only; no code changed.** Line numbers refer to the `CarbonCast-v3.2` branch as of 2026-11-24
(commit `801f1439fbf`). This expands `CONCORDANCE_INTEGRATION.md` sections 4–6 for the second tier (the CNN-LSTM
that forecasts carbon intensity) only. For background on the two tiers, see `HOW_CARBONCAST_WORKS.md`.

## 0. How it makes the model more concordant

The selection rule is the same as in SARIMAX: **among parameter settings that stay within the accuracy budget, pick
the one whose practice forecasts are most concordant.** What changes is which parameters are searched and how
candidates are generated.

### 0.1 The idea

MSE training lands the model at the bottom of a valley. Around that bottom is a set of other weight settings that are
almost as accurate (MSE within 1+eps of the baseline) but order the hours differently. MSE does not care which of
these it picks. Concordance does, so the method searches that near-equally-accurate set for the one that ranks hours
best. SARIMAX does the same thing in log-likelihood terms: the region within delta nats of the MLE, searched for the
most concordant point.

### 0.2 Why an MSE-trained model leaves room

MSE penalises the size of errors, not their order. An MSE-optimal forecast tends to hedge toward the mean. When the
model is unsure whether tomorrow evening's peak will be high or low, the cheapest choice under MSE is a flattened
middle value. That flattening is exactly what causes misordered pairs: two hours that differ in reality get forecast
nearly equal, or in the wrong order. Concordance rewards getting the order right even if the levels are slightly off.
Typical examples are a sharper diurnal contrast, or correct relative levels between days.

### 0.3 How the optimisation moves the model (SLSQP on the output layer, one eps)

1. **Start from the baseline weights.** Compute the baseline training MSE and the practice c-index on the 173
   training windows.
2. **Each pair of hours pushes the weights.** For a pair where actual hour j is higher than actual hour i, the smooth
   term is `sigmoid((f_j − f_i) / (0.1·std(f)))`, and its gradient says to raise f_j relative to f_i.
   - Nearly tied or slightly wrong pairs push hardest, because that is where the sigmoid is steepest.
   - Pairs already clearly right, or hopelessly wrong, barely push.
   - Through the output layer, the push turns into a change in how the LSTM's 24 features map to the 24 hourly
     values. For example, it can amplify the part that produces the daily shape, or shift day-to-day levels.
3. **The MSE constraint limits the move.** SLSQP steps toward higher concordance until the training MSE reaches
   (1+eps) of the baseline, then slides along that boundary. It can still trade one kind of error for another while
   improving the order.
4. **Stretching the forecast cannot game it.** Dividing by std(f) means scaling forecasts up does not raise the
   score; only reordering does.
5. **Pick the answer, as SARIMAX does.** Record every iterate from both starts (the baseline and the previous eps's
   answer). The answer for that eps is the most concordant iterate that satisfies the constraint (`best_within`).
6. **Repeat for each eps,** giving one candidate model per budget. That traces the frontier of concordance gained
   against MSE given up.
7. **Choose eps on the validation half-year,** where the gain actually carries over. Then run that model once on the
   test half-year.

The switching-gradient fallback (section 5.2) walks through the full network's weights instead. It climbs concordance
while feasible and descends MSE when it is not. The final pick is the same rule: the most concordant checkpoint that
satisfied the constraint.

### 0.4 Where it differs from SARIMAX

| | SARIMAX | CarbonCast tier 2 |
|---|---|---|
| What is searched | all 11 interpretable parameters | the 600 output weights (or all ~17k weights) |
| How candidates are generated | SLSQP with finite-difference gradients | SLSQP with exact autodiff gradients, or switching gradient |
| Selection rule | most concordant pooled point within delta | **same**, within eps |
| How long the chosen parameters are used | one forecast week (refit every date) | the whole 6-month test period |
| What a parameter change means | e.g. stronger seasonal AR | a different mapping from learned features to hourly CI |

The last two rows are where the risk lies. In SARIMAX the practice windows sit right before the forecast week. Here,
the weights chosen on the January–June training windows must keep ordering well through the next six months. That is
why eps is chosen on validation and gains are compared against the seed-to-seed spread.

## 1. The problem

```
maximise    C_train(theta)                                smooth c-index over the practice windows in the training half-year
subject to  MSE_train(theta) <= (1 + eps) * MSE_train(theta_0)    the loss fit() already minimises, on the same X, y
```

- **theta_0** is the checkpoint the current `trainModel` produces (`secondTierForecasts.py:524-584`). It stays the
  baseline and is unchanged.
- **Both terms are evaluated with `training=False`** (dropout off), so they describe the model that is actually
  deployed.
- **eps grid:** {0, 0.5, 1, 2, 5, 10, 25}%.
- **The objective and the constraint stay separate.** They are never summed or blended into one loss (section 5).

## 2. What is already available

`trainingandValidationPhase` (`secondTierForecasts.py:855-870`) already has everything in scope:

- `X, y = manipulateTrainingDataShape(trainData, 24, 24, wTrainData)` (`:860`):
  - **X**, shape (≈4273, 24, 15). The columns are `[CI, 5 calendar, 9 weather/source forecasts]`. The column order
    is the same as the test input built in `getForecasts` (`:673-684`).
  - **y**, shape (≈4273, 24).
  - Sample `i` covers history hours `i…i+23` and label hours `i+24…i+47`.
- `valX, valY` (`:862`), plus `wTrainData` and `wValData`.

Sample `i = 24d` (hour 0 of day d) is therefore "history = day d → predict day d+1". A 180-day chunk gives 179 such
daily samples and **173 seven-day windows** (w = 0…172), each covering days w+1…w+7.

Before using this indexing, check that `trainData[0]` is hour 0 (the hour sin/cos column), so that `24d` really is
a day boundary.

## 3. Practice forecasts: three options

The constraint and the solver are the same in all three options. Only the forecast that C scores changes.

| | Practice forecast for window w | Pairs per window | Matches test? |
|---|---|---|---|
| **C_24**: within one 24 h prediction | `model(X[24d])` vs `y[24d]` | 276 | No. It scores only the shape within one day. |
| **C_A**: seven teacher-forced 24 h predictions | `concat(model(X[24(w+k)]) for k=0..6)` vs `concat(y[24(w+k)])` | 14,028 | Same 168 h pairs as test, but every day starts from true history. |
| **C_B**: 7-step rollout | the same as `getDayAheadForecasts`, with predictions fed back | 14,028 | Yes, exactly. |

**Recommendation:** build C_A first, because it uses the 24 h predictions the model is trained on and costs almost
nothing. Keep C_B as the faithful check. The risk with C_A is that its cross-day pairs are learned with true history
at every step, which test does not have from day 2 on. The headroom check (section 7, step 3) measures that gap
before you commit to either.

### 3.1 C_A: no new data code

C_A is only an index into the existing `X, y`:

```python
day   = np.arange(0, len(X), 24)                    # hour-0 samples
win   = day[np.arange(173)[:, None] + np.arange(7)] # (173, 7) sample indices
A_act = y[win].reshape(173, 168)                    # actuals per window
F_A   = model(X[win].reshape(-1, 24, 15), training=False)   # (173*7, 24)
F_A   = tf.reshape(F_A, (173, 168))
```

### 3.2 C_B: the test rollout, batched in TF

```python
hist_ci = X[24*w, :, 0:1]                                  # true CI of day w            (173, 24, 1)
cal[k]  = X[24*(w+k), :, 1:6]                              # true calendar, history day of step k
wx[k]   = wTrainData[168*w + 24*k : 168*w + 24*k + 24]     # block w, rows 24k:24k+24 (as at :619)

ci, out = hist_ci, []
for k in range(7):
    yk = model(tf.concat([ci, cal[:, k], wx[:, k]], -1), training=False)   # (173, 24)
    out.append(yk)
    ci = yk[..., None]                                     # predicted CI becomes the next history
F_B = tf.concat(out, 1)                                    # (173, 168)
```

This was checked against `getDayAheadForecasts` (`:594-642`):

- Step k's history carries the true calendar of the day predicted at step k−1, as at `:622-625`.
- The weather comes from the block issued on the origin's history day. At test, `weatherData[j:j+24]` (`:619`) is
  sliced from one 168-row block per origin (`:630-631`).
- Gradients pass through all 7 calls (backpropagation through the recursion). The cost is 7 forward passes on a
  batch of 173.

The `wx` indexing assumes the 168-row block structure, one block per data day. `select_forecast_blocks` in
`run_6month_forecasting.py` guarantees this and raises an error if the forecast file breaks it
(`CONCORDANCE_INTEGRATION.md` section 3, item 2).

### 3.3 Weather in C_A vs C_B

Both options use weather for the same calendar day (day w+k at step k, the history day of that step). They use
different issues of it:

| | Block used at step k | Lead time of that weather |
|---|---|---|
| C_A | block w+k, rows 0:24 (through X) | 0 days, i.e. fresh |
| C_B and test | block w, rows 24k:24k+24 | k days, as at test |

So C_A also sees better weather than test does. That is a second reason to measure the C_A–C_B gap.

## 4. The smooth c-index in TF

This is a port of `CarbonCast-SARIMAX-mat/src_clean/concordance_impl/concordance.py` (`smooth_concordance`).

```python
I, J = np.triu_indices(168, 1)

def smooth_c(F, A_act, tau=0.1):
    s    = np.sign(A_act[:, J] - A_act[:, I])                       # constants
    keep = (s != 0)                                                  # ties in the actuals dropped
    dF   = tf.gather(F, J, axis=1) - tf.gather(F, I, axis=1)
    sd   = tf.math.reduce_std(F, axis=1, keepdims=True)              # ddof=0, same as np.std
    p    = tf.sigmoid(s * dF / (tau * sd))
    per_window = tf.reduce_sum(p * keep, 1) / keep.sum(1)
    return tf.reduce_mean(per_window)                                # mean over windows, as in SARIMAX
```

- **Scaling does not matter.** Min-max scaling (`common.py:75`) is monotone and `sd` cancels the forecast's scale,
  so C can be computed on scaled values.
- **`sd` stays inside the graph,** so the optimizer cannot raise C by stretching the forecast.
- **Drop flat windows.** Windows with no untied pairs should be excluded (SARIMAX returns NaN for them).
- **Verify the port.** Check every window against the numpy `smooth_concordance` to 1e-9 before using it.
- **Memory:** 173 × 14,028 pairs ≈ 2.4M values, which fits easily. For C_24, use `triu_indices(24, 1)` and 179 windows.

## 5. Solving the constrained problem

Neither solver combines C and MSE into one loss.

### 5.1 First solver: SLSQP on the output `Dense(24)` (600 weights)

Freeze every layer below the final `Dense(24)` (`:565`). The penultimate activations
`H = penultimate(X, training=False)` (the LSTM → Dropout output, shape (4273, 24)) are then **fixed**:

- **Constraint:** `MSE(W, b) = mean((H W + b − y)^2)`. This is an exact quadratic, so the feasible set is an
  ellipsoid. That holds for every objective, C_B included, because the constraint only ever sees teacher-forced X.
- **C_A:** `F = H[win] W + b`. This is linear in the weights; H can be precomputed once and no TF graph is needed
  per evaluation.
- **C_B:** steps 1–6 still depend on W through the fed-back CI. It needs the TF rollout with only the last layer
  trainable.

```python
w0   = np.concatenate([W0.ravel(), b0])          # from model.layers[-1].get_weights()
MSE0 = mse(w0)                                   # equals model.evaluate(X, y) with dropout off

res = minimize(lambda w: -C(w), w0, jac=lambda w: -gradC(w), method="SLSQP",
               constraints=[{"type": "ineq",
                             "fun": lambda w: (1 + eps) * MSE0 - mse(w),
                             "jac": lambda w: -grad_mse(w)}],
               callback=record_iterate)
```

- **Gradients** come exactly from TF autodiff over the full data (one batch), so runs are deterministic. The MSE
  gradient also has a closed form: `2 / (N*24) * H^T (H W + b − y)`.
- **Starts and pooling:** run from w0 and from the previous eps's answer, in increasing eps order. Pool every
  iterate and keep the most concordant feasible point (`best_within`), exactly as in `constraint_study.sweep`.
- **Precision:** cast H, y and the weights to float64. Float32 gradients make SLSQP stall.
- **Record the free slack.** theta_0 was trained with dropout and picked on validation loss, so its output layer is
  not least-squares optimal on dropout-off H. The closed-form least-squares point has `MSE_LS <= MSE0`. Record
  `MSE_LS / MSE0`: part of any gain at eps = 0 may come from that slack alone.
- **Write back** with `model.layers[-1].set_weights([W, b])` and save the model as `<region>_eps_<e>.h5`.
- **Limitation:** only the output mapping moves, so the network cannot learn new features.

### 5.2 Fallback solver: switching gradient on the whole network

Use this if 600 weights give too little room (a flat frontier in 5.1). It is Polyak's switching gradient method; the
stochastic version is Lan & Zhou's cooperative stochastic approximation.

```python
optC, optM = Adam(lr_c), Adam(lr_m)       # separate optimizers, so moment estimates don't mix objectives
for step in range(S):
    if step % K == 0:                      # full-data check, dropout off
        r, c = mse_full(model) / MSE0, C_train(model)
        record(step, r, c)
        if r <= 1 + eps:
            keep_if_best(c, model.get_weights())
        mode = "C" if r <= 1 + eps else "MSE"
    if mode == "C":
        loss = -C_train(model)             # all 173 windows (full batch is cheap)
        opt = optC
    else:
        xb, yb = next_minibatch(X, y)
        loss = mse(model(xb, training=False), yb)
        opt = optM
    opt.apply_gradients(...)
```

- **Start from theta_0,** which is feasible.
- **Each step follows one objective only.** The answer is the best feasible checkpoint, so no multiplier and no
  weighted loss appear anywhere.
- **Only checked points are eligible,** because the model can drift out of the feasible set between checks.
- **Dropout off in both modes,** so the optimizer targets exactly the quantities being evaluated.
- **To implement it,** use either a custom `GradientTape` loop or a `keras.Model` subclass whose `train_step` reads a
  `tf.Variable` mode flag, plus a batch-end callback that sets the flag. Either replaces
  `ModelCheckpoint(val_loss)` (`:572`) for this stage.
- K (for example 20 steps), the two learning rates and S need a small tuning pass. Tune on train only.

## 6. Where it plugs into the code (nothing edited)

| Location | What happens there |
|---|---|
| **Precondition (done):** `secondTierForecasts.py:502` and `select_forecast_blocks` in `run_6month_forecasting.py` | The weather stride and `sort_index` bugs are fixed (`CONCORDANCE_INTEGRATION.md` section 3). Without those fixes, X's weather (C_A) and the `wTrainData` blocks (C_B) would be misaligned. |
| New module `src/concordance_tier2.py` | `build_windows`, `smooth_c`, `rollout_B`, `mse_train`, `solve_output_slsqp`, `switching_gradient`, `best_within` |
| `trainingandValidationPhase`, after the `trainModel` call (`:868`) | Compute MSE_0, run the eps sweep and return `{"baseline": theta_0, "eps_<e>": theta_e, ...}`. Everything needed is already in scope. |
| `trainModel`, `:524-584` | Unchanged. The existing `-s` flag loads the saved `<region>.h5` (`:535-538`), so the sweep can run on baselines you already have, with no retraining. |
| Main loop in `runSecondTier`, `:177-190` | Loop over the variants. Call the **unchanged** `getDayAheadForecasts` on test for each one and write one CI forecast CSV per variant. |
| `getScores`, `:788-826` | Add the 168 h-window c-index, both lifelines and smooth, so it matches SARIMAX's forecast-week c-index. |
| `secondTierConfig.json` | Add a `CONCORDANCE` block: `OBJECTIVE` (`C24`, `A` or `B`), `EPS`, `TAU`, `SOLVER` (`slsqp_output` or `switching`), `SEEDS`. |
| `run_6month_forecasting.py`: `create_chunk_config`, `merge_forecast_results` | Pass the settings through and merge per variant. |

**Validation and carry-over.** Build the same windows from `valX, valY, wValData` (Jul–Dec 2022 for H1). Use them
only to measure each variant and to choose eps, never inside the solver.

## 7. Order of work

1. **Fix the two alignment bugs (done).** Both are fixed, and every training, validation and test input now gets
   weather for its own hours. Re-check this after any change to the weather data or the chunk dates.
2. **Port `smooth_c`** and test it against the numpy version.
3. **Headroom check on the baseline.** Compute C_24, C_A and C_B on train and val, plus the hard (lifelines) c-index
   of the real test rollouts. A large C_A–C_B gap means C_A is not a safe stand-in for the test rollout.
4. **Output-layer SLSQP with C_A** over the eps grid. This should take minutes per chunk.
5. **Repeat with C_B.** Try the switching gradient if the frontier is flat.
6. **Choose eps on val, then test once.** Run 3–5 seeds. Count a gain only if it beats the baseline's seed-to-seed
   spread. Report for every variant:
   - the MSE_train ratio, and `MSE_LS / MSE0`;
   - train C;
   - val C and val MSE;
   - test 168 h c-index, MAE and MAPE.

   Feed the per-variant CSVs into the scheduling code as described in `CONCORDANCE_INTEGRATION.md` section 7.
