# How CarbonCast works, compared with the SARIMAX study

This was written from `src/firstTierForecasts.py`, `src/secondTierForecasts.py` and the two config files on the
`CarbonCast-v3.2` branch (2026-11-21). Line numbers refer to that version. It is background for
`CONCORDANCE_INTEGRATION.md`.

## 1. The big picture

```
                    TIER 1 (one small ANN per source)              TIER 2 (one CNN-LSTM per region)
                    ---------------------------------              --------------------------------
coal history  ──►  ANN_coal  ──► 168 h coal forecast   ─┐
gas history   ──►  ANN_gas   ──► 168 h gas forecast    ─┤
wind history  ┐                                         ├─► forecasts file ─┐
+ weather fcst┴►  ANN_wind  ──► 168 h wind forecast    ─┘   (weather +      │
                                                              source fcsts) ├─► CNN-LSTM ──► 168 h CI forecast
CI history (computed from actual mix × emission factors) ───────────────────┘
```

Carbon intensity (CI) is **never computed from the source forecasts with a formula at forecast time**. The historical
CI target is computed from the actual generation mix and emission factors (`src/carbonIntensityCalculator.py`). The
second tier then *learns* to predict CI, with the source forecasts as input features.

## 2. Tier 1: source production forecasts (`src/firstTierForecasts.py`)

- **One model per source per region.** AECI has three: coal, natural gas and wind (`firstTierConfig.json`, `SOURCES`).
- **Model:** a feed-forward ANN, Flatten → Dense(50) → Dense(34) → Dense(24), with MSE loss and Adam (`:640-666`).
- **Input**, a 24 h window:
  - that source's own last 24 h of generation, plus 5 calendar features (hour sin/cos, month sin/cos, weekend);
  - **only for renewables** (solar, wind, hydro), 5 weather-forecast features for the same 24 h: wind speed,
    temperature, dewpoint, solar radiation and precipitation (`:84-87`, `NUM_FEATURES_PER_SOURCE`).
  - Coal, gas, nuclear and oil get **no weather at all**: they are purely autoregressive, plus calendar.
  - A source never sees the other sources or demand.
- **Output:** the next 24 h of that source.
- **Training:** windows sliding by 1 hour, always with real history as input. A `ModelCheckpoint` on the last 30 days'
  validation loss picks the model (`NUM_VAL_DAYS: 30`).
- **Periods** (`TRAIN_TEST_PERIOD`): `PERIOD_0` trains on 2019–2021 and forecasts 2022; `PERIOD_1` trains on 2020–2022
  and forecasts 2023 (three training years each, rows `[DATASET_START, DATASET_LIMITER)`). Both sets of forecasts are out of sample.
- **168 h forecast:** each day, the 24 h model is chained 7 times with its own predictions fed back in
  (`getDayAheadForecasts`, `:668-712`), then the next day restarts from actual data (section 4). For CISO, the
  published day-ahead solar/wind forecasts replace day 1 when available (`PARTIAL_FORECAST_AVAILABILITY_LIST`).
- **Output file:** all sources' 168 h forecasts, stacked with the weather, form `data/<REGION>/<REGION>_168hr_forecasts_DA.csv`
  (`aggregateDataAndGenerateForecastFile`, `:369`). It holds one 168-row block per issue day. AECI's file has 122,136
  rows but only 17,592 distinct timestamps, because the blocks overlap.

## 3. Tier 2: carbon intensity forecast (`src/secondTierForecasts.py`)

- **One model per region**, one for direct emission factors (`-d`) and one for lifecycle (`-l`).
- **Model:** a CNN-LSTM, Conv1D(4) → MaxPool → Conv1D(16) → Flatten → RepeatVector(24) → LSTM(24) → Dropout → Dense(24),
  with MSE loss and Adam (`:554-569`). For AECI it has about 17,340 weights.
- **Input**, a 24 h window with these features per hour:
  - the last 24 h of **actual CI**, plus the same 5 calendar features (`NUM_FEATURES: 6`);
  - the 5 weather-forecast features and **tier 1's source forecasts** for the matching hours. For AECI that is coal,
    gas and wind (`NUM_FORECAST_FEATURES: 9`), so 15 features in total.
  - Actual source generation history is *not* an input; only CI history is.
- **Output:** the next 24 h of CI.
- **Training** follows the 6-month chunks (`scripts/run_6month_forecasting.py`). The model is trained **once per
  chunk**, then forecasts every test day with fixed weights.

  | Chunk | Train | Val | Test |
  |---|---|---|---|
  | H1 | Jan–Jun 2022 (181 days) | Jul–Dec 2022 (184) | Jan–Jun 2023 (181) |
  | H2 | Jul–Dec 2022 (184) | Jan–Jun 2023 (181) | Jul–Dec 2023 (184) |

  "Test" means the days the test forecasts start on, one forecast per day (`CHUNKS` in the script). The last
  forecast runs 6 days past the test period, so each chunk's data extends 6 days beyond it. If a region's data stops
  too early, the script ends the test period at the last day the data supports and prints a warning. AECI's H2 test
  period ends on Dec 28, because its emissions end on 2024-01-03 and its last forecast block is from 2023-12-28.

  Training windows slide by 1 hour with 24 h labels (`manipulateTrainingDataShape`, `:474-508`). A `ModelCheckpoint`
  on validation MSE picks the model (`:571-580`). `EPOCH` is 10 and early-stopping patience is also 10, so early
  stopping never fires.
- **Tier 1's forecasts in tier 2's training data are out of sample.** The 2022 inputs came from tier 1's 2019–2021
  model, so tier 2 learns how far to trust them, errors included.
- **168 h forecast:** the same 7-step chained rollout as tier 1 (`getDayAheadForecasts`, `:594-642`).

## 4. How one 168 h forecast is produced

Within a single 168 h forecast, the model **never sees real CI after the origin**. In `getDayAheadForecasts`
(`secondTierForecasts.py:609-631`):

```python
for i in range(num_test_days):              # one 168 h forecast per day
    tempHistory = history.copy()            # true history up to the forecast start
    for j in range(0, 168, 24):             # 7 steps of 24 h
        yhat = getForecasts(model, tempHistory, ..., weatherData[j:j+24])
        latestHistory = testData[currentDayHours+j : currentDayHours+j+24].tolist()
        for k in range(24):
            latestHistory[k][0] = yhat[k]   # CI column replaced by the prediction
        tempHistory.extend(latestHistory)
    history.extend(testData[currentDayHours : currentDayHours+24].tolist())  # real day added
```

- **Within one forecast (steps 1–7).** Step 2's CI input is step 1's 24 predictions, and so on through hour 168, so
  errors compound. The calendar features in the history are the true values, since they're known in advance. Each step
  also gets the matching 24 h slice of that day's weather and source-forecast block.
- **Between forecasts (the next day).** The real CI for the day that just passed is appended to `history`, so the next
  forecast starts from actual data. Nothing carries over between forecasts except that real history.
- **Training never does this.** Every training input is actual CI (teacher forcing), and every label is the next 24 h
  of actual CI. The model never sees its own predictions as input during training. Steps 2–7 only ever happen at test
  time, on inputs it was never trained on.

Tier 1 works the same way, with source generation in place of CI.

## 5. Where the SARIMAX picture holds and where it breaks

| | SARIMAX study | CarbonCast |
|---|---|---|
| Forecast setup | 168 h from an origin, using forecast weather | **same** |
| What is forecast | each source, then combined into CI by emission-factor formula | tier 1: each source. Tier 2: **CI directly**, learned with source forecasts as inputs; no combine formula |
| Fitting frequency | refitted for **every forecast date** on the previous 504 h | trained **once per 6-month chunk** (181 or 184 days), then reused for 181–184 test days |
| Parameters | 11 interpretable ones | about 17k weights (tier 2); about 9.8k per fossil source and 15.8k per renewable source (tier 1); stochastic training |
| "Best fit" | a unique MLE that the optimizer converges to | a checkpoint picked by validation loss; a different seed gives a different model |
| Memory | Kalman state summarises the whole history | only sees the last 24 h window |
| One step | 1 hour (Kalman one-step loglike) | a 24 h block |
| Multi-step forecast | the model's natural mode: its dynamics roll forward | 7 chained calls of a 24 h model that was **never trained chained** |
| Weather | linear regression terms | non-linear features; tier 1 fossil sources get none |

## 6. What this means for concordance

- **The objective/constraint form transfers fine.** Maximise practice concordance subject to an accuracy floor on the
  training loss (`CONCORDANCE_INTEGRATION.md`, section 4).
- **"Practice windows" means something different.** SARIMAX had 4 windows inside a 3-week fit. CarbonCast has
  174–177 daily windows across a 6-month training period, and the weights they produce must work for the next 6 months of
  tests. The carry-over question is seasonal, not next-week.
- **Tier 2 is where CI ranking lives.** The SARIMAX finding that per-source gains didn't reach the CI ranking may not
  apply here, because there's no combine step between the optimised model and the CI forecast.
- **The concordance gradient passes through the 7-step chain.** That is the main mechanical difference from SARIMAX,
  and the only training signal the network would ever get about its multi-step behaviour.
- **Tier 1 fossil sources have no weather input.** Concordance there can only come from their own recent history and
  calendar patterns. That is another reason to start with tier 2, or with wind in tier 1.
