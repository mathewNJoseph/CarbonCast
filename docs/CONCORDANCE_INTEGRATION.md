# Adding the concordance objective to CarbonCast

Status: **observations and plan only; no code changed.** Line numbers refer to the `CarbonCast-v3.2` branch as of
2026-11-21. For how the two tiers work and how they differ from the SARIMAX setup, see `HOW_CARBONCAST_WORKS.md`. For the detailed second-tier design (practice-window options, the TF c-index,
solvers and code hooks), see `TIER2_CONCORDANCE_DESIGN.md`.

## 1. What is being carried over from SARIMAX

The SARIMAX study (`CarbonCast-SARIMAX-mat/src_clean/constraint_study/01_constraint_method.md`) fits

    maximise   C(theta)                            mean smooth c-index over the practice windows
    subject to loglike(theta) >= loglike(theta_MLE) - delta     delta in nats

| Piece | SARIMAX implementation |
|---|---|
| Model | one SARIMAX(1,0,1)x(1,1,1,24) per source, refitted for every forecast date on 504 h |
| Practice windows | origins 168, 224, 280, 336 h into the training window; each a 168 h multi-step forecast made with the forecast weather, scored against the training actuals (`concordance_impl/source_model.py`, `practice_forecasts`) |
| C(theta) | `smooth_concordance`: mean over pairs of `sigmoid(sign(a_j - a_i) * (f_j - f_i) / (tau * std(f)))`, tau = 0.1, ties in the actuals dropped, averaged over windows (`concordance_impl/concordance.py`) |
| loglike | Kalman one-step predictive Gaussian log-likelihood, about 454 terms |
| Reference point | the MLE (L-BFGS, then a Powell polish) |
| Budget | delta grid {0, 1, 2, 5, 20, 50} plus the LR budget ½·χ²(k, 0.95) = 9.84 for k = 11 |
| Solver | SLSQP with finite-difference gradients, two starts (the MLE and the previous delta's answer); the answer is the most concordant pooled point inside the budget (`best_within`) |

The goal is the same structure for CarbonCast: the same C, practice forecasts from the training period, and an
accuracy floor. The floor does not have to be a log-likelihood. It is the loss the CNN-LSTM already minimises, its
training MSE, with a relative tolerance eps in place of delta (section 4).

## 2. What CarbonCast looks like, as far as this matters

### 2.1 Model and training
- **Two tiers, each predicting 24 h at a time.** The first tier is a small per-source ANN
  (`src/firstTierForecasts.py:640-666`, Dense 50 → 34 → 24). The second tier is the CNN-LSTM that predicts carbon
  intensity (`src/secondTierForecasts.py:554-565`: Conv1D(4) → MaxPool → Conv1D(16) → Flatten → RepeatVector(24) →
  LSTM(24) → Dropout → Dense(24)).
- **Training never sees a 168 h horizon.** `trainingandValidationPhase` builds labels with
  `manipulateTrainingDataShape(trainData, 24, 24, ...)` (`secondTierForecasts.py:860-862`), so each sample is
  "24 h of history plus 24 h of weather and source forecasts → the next 24 h", with origins sliding hourly.
- **The 168 h forecast is a recursive rollout.** `getDayAheadForecasts` (`secondTierForecasts.py:594-642`) calls the
  24 h model 7 times. After each call it puts the predicted CI into the history column and keeps the true date-time
  features (lines 615-625). Each origin then extends the history with the *real* day (line 628).
- **The loss is MSE** (`secondTierConfig.json:58`, compiled at `secondTierForecasts.py:569`). That is the accuracy
  measure the constraint uses (section 4).
- **Model selection uses validation MSE only**: `ModelCheckpoint(monitor='val_loss', save_best_only=True)` and the
  best checkpoint is reloaded (`secondTierForecasts.py:571-580`). `EPOCH` is 10 and early-stopping patience is also
  10, so early stopping never fires and only the checkpoint does any selecting.
- **Inputs are min-max scaled with training statistics** (`common.py:75`). The smooth c-index divides by the forecast's
  own std, so it is invariant to this scaling and can be computed on scaled values.
- **One run per region and no fixed seed** (`NUMBER_OF_EXPERIMENTS_PER_REGION: 1`). SARIMAX fits were bit-identical
  across nodes. CarbonCast fits are not, and that changes how results have to be compared (section 7).

### 2.2 Data periods (6-month chunks, `scripts/run_6month_forecasting.py`)
| Chunk | Train | Val | Test |
|---|---|---|---|
| H1 | Jan–Jun 2022 (~180 days) | Jul–Dec 2022 (180) | Jan–Jun 2023 (180 + 6 buffer) |
| H2 | Jul–Dec 2022 | Jan–Jun 2023 | Jul–Dec 2023 |

There are about 4,300 training hours, which allows about 173 daily 168 h practice origins in the training period.
SARIMAX had 4.

### 2.3 Weather layout
`data/<REGION>/<REGION>_168hr_forecasts_DA.csv` is one 168-row block per issue day: the AECI block issued on
2022-01-01 runs to 2022-01-07 23:00, and the next block starts at 2022-01-02 00:00. AECI has 122,136 rows but only
17,592 distinct timestamps, so most hours appear 7 times. Test-time code steps through the blocks 168 rows at a time
(`secondTierForecasts.py:630-631`).

## 3. Weather alignment (issues 1 and 2 are fixed)

Issues 1 and 2 would have made any practice forecast meaningless. Both are now fixed.
With the fixes, every training, validation and test input in both chunks gets weather
for exactly its own hours (checked on the AECI data). Before them, that held for 0% of inputs.

1. **Second-tier training weather stride (fixed).** `manipulateTrainingDataShape` added `weatherIdx += 23` at each
   day boundary (`secondTierForecasts.py:502`), so each day advanced 47 rows, not 168. Day *d*'s training samples read
   weather from block ≈ 0.28·*d*. The line was changed in commit `84e74bb` and has been restored to
   `weatherIdx += (MAX_PREDICTION_WINDOW_HOURS-24)`, matching the first tier and test time.
2. **`sort_index()` on the overlapping forecast file (fixed).** `modify_data_for_chunk` sorted the forecast DataFrame
   by timestamp and sliced it by row timestamp, so every 168 rows were no longer one issue's 7-day block.
   `select_forecast_blocks` (`run_6month_forecasting.py:67`) now keeps whole blocks in file order. It keeps blocks
   issued from the chunk's first day through its last day minus 6, which is one block per data day as
   `splitWeatherDataset` and both rollouts assume. It raises an error if the file has a missing day or a broken block,
   for example after the weather aggregation drops a cycle.
3. **Which day's weather is paired with which input (not a bug).** In both training and the rollout, block *b* is the
   forecast issued on data day *b*, and step *j* uses block rows `j*24 : j*24+24`. Those rows cover the *history*
   window's hours, not the day being predicted. This is consistent between train and test, so the practice rollout has
   to reproduce it exactly.

## 4. The accuracy constraint

**Definition.** Constrain the loss the model already minimises: the training MSE that `model.fit` uses.

    maximise   C(theta)
    subject to MSE_train(theta) <= (1 + eps) * MSE_train(theta_0)

theta_0 is the baseline checkpoint and eps is a relative tolerance. MSE_train is computed on exactly the samples
`fit` trains on: the `X, y` from `manipulateTrainingDataShape` (`secondTierForecasts.py:860`), with hourly-sliding
origins, 24 h labels and scaled units. Evaluate it with dropout off and fixed weights (`model.evaluate(X, y)`), not
from the loss Keras reports during an epoch. That number includes dropout and changes as the weights update.

**Why this form.**
- "Accuracy given up" is measured in the model's own objective, with no distributional assumption.
- The constraint is a ratio on the same sample set, so the overlapping hourly windows and the sample count drop out.
- Objective and constraint stay separate (no blend), so nothing is added across units.

**Budget grid.** Use eps in {0, 0.5%, 1%, 2%, 5%, 10%, 25%}. For a rough comparison, SARIMAX's nats budgets are
about these MSE increases under a Gaussian reading of its log-likelihood (exp(2·delta/454) − 1 on its ~454 terms):

| SARIMAX delta (nats) | ≈ MSE increase |
|---|---|
| 1 | 0.4% |
| 5 | 2.2% |
| 9.84 (LR) | 4.4% |
| 20 | 9.2% |
| 50 | 24.6% |

The grid brackets the SARIMAX sweep. The match is approximate: the Kalman log-likelihood weights hours by a varying
F_t and scores 1 h-ahead errors, while CarbonCast scores 24 h-ahead ones. The likelihood-ratio budget has no
counterpart here and is dropped. With about 17,340 parameters it would mean nothing anyway.

**theta_0 is not a minimum of MSE_train.** The baseline is the checkpoint with the best *validation* loss, trained
with dropout, so some points with a *lower* training MSE may exist and are feasible. Record MSE_train(theta_0) and
each variant's ratio to it in every table.

**Train, not val.** Constrain the training MSE, the counterpart of SARIMAX's in-sample log-likelihood. That keeps the
validation half-year clean for choosing eps and measuring carry-over. Report validation MSE and test MAPE per eps
alongside it.

## 5. The concordance (the objective)

**Practice forecasts.** These are the training-period copy of `getDayAheadForecasts`. For each daily origin *d* in the
training period, with 24 h of history before it and 168 h of actuals after it:
1. start from the *true* 24 h history (CI plus the 5 date-time features) ending at *d*;
2. for *j* = 0…6, call the model on `[history, weather-block(d) rows j*24 : j*24+24]` and write the predicted CI into
   the next 24 h of history, keeping the true date-time features, exactly as at `secondTierForecasts.py:618-625`;
3. score the 168 predictions against the 168 actuals with the smooth c-index (tau = 0.1, std-normalised), then
   average over origins.

Skip origins whose 168 actuals are flat, as SARIMAX's `scored` mask does.

**It must be differentiable.** `getForecasts` uses `model.predict` per call (`secondTierForecasts.py:681`), and
`getDayAheadForecasts` is a Python loop over numpy lists, so neither carries gradients. The practice rollout should be
written in TensorFlow: batch all origins (about 173) and call `model(x, training=False)` 7 times in sequence inside a
`GradientTape`, concatenating the predicted CI into the next input. This is backpropagation through the 7-step
recursion, the analogue of SARIMAX filtering through NaN endog. It is cheap: 7 forward passes on a batch of about 173.

**Port `smooth_concordance` to TF.** Use `triu_indices` over 168 → 14,028 pairs per window, mask the ties in the
actuals, and keep `std(f)` inside the graph so the optimizer cannot gain by stretching the forecast.

**The rollout meets something new here.** SARIMAX's MLE already encodes multi-step dynamics. The CNN-LSTM was only
ever fitted to 24 h, so the concordance term is the first multi-step training signal it gets. That can help more than
it did for SARIMAX. It can also be less stable, since errors compound through the 7 steps.

## 6. Where it goes in the code

### 6.1 Second tier (recommended first: it ranks carbon intensity directly, which is what scheduling uses)

| Location | Today | Change for concordance |
|---|---|---|
| `runSecondTier`, `secondTierForecasts.py:109-160` | split, fill and scale train/val/test and weather | build the practice-window tensors here from `trainData` and `wTrainData` after scaling: per origin, the true history, the 7 weather slices, the 168 actuals and a `scored` flag. The constraint reuses the training `X, y` that `fit` already gets. |
| `trainingandValidationPhase`, `:855-870` | one `trainModel` call | stage 1: train the baseline as now and record MSE_train(theta_0). Stage 2: for each eps in increasing order, run the constrained fine-tune from the baseline and from the previous eps's answer, and keep `best_within` (as in `constraint_study.sweep`). |
| `trainModel`, `:524-584` | Keras `fit` on MSE; checkpoint on `val_loss`; reloads `<region>.h5` | the baseline is unchanged. The constrained stage needs a custom loop (`GradientTape`, or a `keras.Model` subclass with its own `train_step`), because the objective is a 7-call rollout over whole windows and does not fit `fit(X, y)` batches. Save one model per eps (`<region>_eps_<e>.h5`); `:572` currently overwrites a single file. |
| `getDayAheadForecasts`, `:594-642` | numpy rollout with `predict` | reuse its indexing as the specification for the TF practice rollout. Keep this function for test so the test path does not change. |
| `getScores`, `:788-826` | RMSE, MAPE, daily MAPE | add the 168 h-window c-index (lifelines and smooth) on test, and on val for carry-over, so the result matches SARIMAX's "forecast-week c-index". |
| main loop, `:177-244` | one CSV per experiment | loop over variants (`baseline`, `eps_<e>`) and write one CI forecast file per variant. These files feed `CarbonCast-SARIMAX-mat`'s R concordance and scheduling figures. |
| `secondTierConfig.json` | `SECOND_TIER_CNN_LSTM_MODEL_HYPERPARAMS` | add a `CONCORDANCE` block: `EPS`, `TAU`, `PRACTICE_ORIGIN_STRIDE_DAYS`, `PRACTICE_PERIOD` (train), solver settings and `SEEDS`. |
| `run_6month_forecasting.py:26-60, 202` | per-chunk config and merge | pass the eps settings through, and merge per variant. |

### 6.2 Solving the constrained problem
SLSQP is not usable here. It builds a dense quasi-Newton matrix (17,340² ≈ 300M entries), and finite differences
would cost 17k rollouts per gradient. Options, from cheapest to most faithful:

1. **Selection only (no new training objective).** Replace the `val_loss` checkpoint with a callback that evaluates
   (training MSE, practice C) at every epoch, over several seeds and learning rates. The answer for each eps is the
   most concordant checkpoint within the floor. This is SARIMAX's `best_within` over a pool of points. It takes
   a few hours to build and shows whether the frontier exists at all.
2. **SLSQP on the output layer (run first; the SARIMAX formulation unchanged).** Freeze everything except the final
   Dense layer: 600 weights in tier 2, 840 in tier 1. Pass objective and constraint to scipy separately:
   `fun = -C_train(w)` and `constraints = [{"type": "ineq", "fun": (1+eps)*MSE_0 - MSE_train(w)}]`. Both gradients
   come exactly from TF autodiff over the full training set (about 4,300 samples in one batch), so runs are
   deterministic. Use the same two starts and `best_within` pooling as `constraint_study.sweep`. Limitation: only the
   output mapping moves, so the network cannot learn new features.
3. **Switching gradient (whole network, if the output layer has too little room).** This is Polyak's switching
   gradient method; its stochastic version is Lan & Zhou's cooperative stochastic approximation. Start from the
   baseline, which is feasible. Every K steps (for example, 20 batches), evaluate the full training MSE with dropout
   off. If `MSE_train <= (1+eps)*MSE_0`, the next K steps do gradient ascent on C only; otherwise they do gradient
   descent on MSE only, until it is feasible again. Record (MSE_train, C) at every check, and keep the most concordant
   feasible checkpoint (`best_within`). Each step follows one objective and C and MSE are never summed, so there is
   no multiplier or weighted loss. To implement it, use a `keras.Model` subclass whose `train_step` reads a
   `tf.Variable` flag, plus a batch-end callback that sets the flag and replaces `ModelCheckpoint(val_loss)`.

Do not combine C and MSE into one loss (for example `-C + lam*g`). The formulation keeps the objective and the
constraint separate.

For every option, evaluate C and MSE_train with dropout off, so that the constraint is checked on the model that is
actually deployed.

### 6.3 First tier (second experiment)
The per-source ANN (`firstTierForecasts.py:588-712`) is the closer analogue of SARIMAX's per-source fits. In the
SARIMAX dev run, wind was the only source whose multi-step gain carried over to the forecast week. A first-tier,
wind-only constrained fit is therefore the natural follow-up. It costs more: the first tier writes
`<REGION>_168hr_forecasts_DA.csv`, so the second tier has to be retrained on the new source forecasts afterwards.

## 7. Evaluation, to match the SARIMAX study

- **Practice C vs training MSE given up (%)**, one point per eps: the frontier figure (`plot_constraint.fig_frontier`,
  with the x-axis in % MSE instead of nats).
- **Carry-over.** Use the validation half-year (2022) to choose eps, as the SARIMAX dev run did, and test 2023 only
  once. Report the 168 h CI c-index, MAE and MAPE change from the baseline, with an interval from resampling whole
  origin dates.
- **Seeds.** Unlike SARIMAX, a refit moves the answer by itself. Run at least 3 to 5 seeds of the baseline, and of
  each eps from each baseline. A gain only counts if it beats the seed-to-seed spread of the baseline.
- **Scheduling.** Feed the per-variant CI forecast CSVs into the existing extra-carbon and WRR code
  (`scheduling_study`), so the results can be compared directly with the SARIMAX tables.

## 8. Open questions
- 168 h pairs mostly reward getting the diurnal cycle right. SARIMAX has the same limitation, but a per-day (24 h)
  c-index may suit 24 h scheduling windows better. Decide this before the sweep, not after.
- Should practice windows also use the validation period? That gives more windows, but then the validation period can
  no longer be used to choose eps.
- Should the constraint be the 24 h MSE the model trains on (the default here), or the MSE of the 168 h practice
  rollouts? The second measures forecast-week accuracy more directly, but the model never optimises it by default.
