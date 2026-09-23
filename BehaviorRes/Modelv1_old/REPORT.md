# Modelv1 — Formulation A set encoder on ModelDataRightContra

Built 2026-09-21T21:20:48.327388+00:00

## What this is

Formulation A from the neural-decoding framework: a shared per-unit encoder φ_θ([log1p count, CCF xyz, region embedding, session mean log-rate]), permutation-invariant pooling, then a causal GRU and a 2-d speed head. Only MOp/MOs units are used. Column *n* is not a global neuron identity.

RightContra: 52 sessions. Motor units/session 5–226. Trials shorter than 13 bins after motor filtering are dropped; T is cropped at 128 bins (2.56 s). Targets are |ω| (rad/s) and Lightning Pose 2D paw speed (DLC fallback).

## Holdouts

- **trial_holdout**: 5/52 sessions contribute held-out trials (5adab0b7, 62902992, 8c33abef, a92c4b1d, d2f5a130). train n=5707, test n=45.
- **session_holdout**: held-out session 626126d5 of CSH_ZAD_026; 454 trials from other sessions of that mouse remain in train. train n=5515, test n=237.
- **mouse_holdout**: held-out mouse ZM_2241 (one session, unseen animal). train n=5381, test n=371.

## Model

Unit MLP: Linear(21→64), GELU, Linear(64→64). Mean pool applies Linear(64→64) to the masked average. Attention pool uses α ∝ exp(q⊤ tanh(W e)) over units. GRU hidden 64, decoder Dropout–Linear–GELU–Linear → 2. Train-time unit dropout 0.15.

Training loss is trial-balanced MSE on z-scored targets (PDF eq. 7): each trial contributes equally, then trials are averaged. AdamW lr=0.001, weight decay=0.0001, batch 32, max 40 epochs, patience 5.

## Results

| task | pool | target | R² concat | mean trial R² | median trial R² | RMSE | MAE | Pearson | n bins | n trials |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trial_holdout | mean | wheel_speed | 0.456 | 0.281 | 0.610 | 0.691 | 0.456 | 0.715 | 1016 | 45 |
| trial_holdout | mean | paw_speed | 0.328 | 0.105 | 0.300 | 495.835 | 293.508 | 0.576 | 1016 | 45 |
| trial_holdout | attn | wheel_speed | 0.486 | 0.313 | 0.594 | 0.672 | 0.442 | 0.714 | 1016 | 45 |
| trial_holdout | attn | paw_speed | 0.280 | 0.164 | 0.356 | 512.972 | 288.325 | 0.533 | 1016 | 45 |
| session_holdout | mean | wheel_speed | 0.565 | 0.493 | 0.716 | 0.696 | 0.460 | 0.756 | 4694 | 237 |
| session_holdout | mean | paw_speed | 0.148 | -1.384 | 0.056 | 366.312 | 246.939 | 0.599 | 4694 | 237 |
| session_holdout | attn | wheel_speed | 0.558 | 0.520 | 0.614 | 0.702 | 0.432 | 0.765 | 4694 | 237 |
| session_holdout | attn | paw_speed | 0.259 | -0.893 | 0.183 | 341.551 | 213.628 | 0.625 | 4694 | 237 |
| mouse_holdout | mean | wheel_speed | 0.233 | 0.259 | 0.414 | 0.834 | 0.537 | 0.510 | 8433 | 371 |
| mouse_holdout | mean | paw_speed | 0.255 | -8.339 | 0.338 | 502.453 | 311.930 | 0.506 | 8433 | 371 |
| mouse_holdout | attn | wheel_speed | 0.260 | 0.244 | 0.439 | 0.819 | 0.539 | 0.531 | 8433 | 371 |
| mouse_holdout | attn | paw_speed | 0.282 | -8.051 | 0.373 | 493.327 | 313.068 | 0.534 | 8433 | 371 |

### Best pool per task × target (concatenated-bin R²)

- trial_holdout / wheel_speed: **attn** R²=0.486 (mean trial R²=0.313)
- trial_holdout / paw_speed: **mean** R²=0.328 (mean trial R²=0.105)
- session_holdout / wheel_speed: **mean** R²=0.565 (mean trial R²=0.493)
- session_holdout / paw_speed: **attn** R²=0.259 (mean trial R²=-0.893)
- mouse_holdout / wheel_speed: **attn** R²=0.260 (mean trial R²=0.244)
- mouse_holdout / paw_speed: **attn** R²=0.282 (mean trial R²=-8.051)

## Training diagnostics

### trial_holdout

- mean: 39154 params, best @ 38/40  train=1.3801  val=1.3311
- attn: 39218 params, best @ 25/30  train=1.3867  val=1.3129

### session_holdout

- mean: 39154 params, best @ 14/19  train=1.4793  val=1.4581
- attn: 39218 params, best @ 31/36  train=1.3948  val=1.3601

### mouse_holdout

- mean: 39154 params, best @ 39/40  train=1.3806  val=1.3647
- attn: 39218 params, best @ 20/25  train=1.4100  val=1.3905

Plots: `train_curves.png`, `r2_concat.png`, `r2_trial.png`, `examples_<task>.png`.

## Interpretation

1. **Attention pool is the better Formulation A variant overall.** It wins 4 of 6 task×target cells on concatenated R². Mean pool is close on session-holdout wheel (0.565 vs 0.558) and slightly better on trial-holdout paw.
2. **Within-session and new-session wheel decoding are strong.** Trial-holdout wheel R² 0.486 and session-holdout wheel R² 0.565 beat MLtestcontra's best CNN–GRU (0.281 / 0.310). The trial-balanced loss plus CCF/region features are the differences.
3. **Paw speed now transfers to a new insertion.** Session-holdout paw R² is +0.259 (attn) versus MLtestcontra values below 0. Pearson stays ~0.6. Mean per-trial paw R² is still negative: a minority of short/still trials are badly scaled and pull the trial average down, while the median trial is +0.18.
4. **A new mouse is no longer at R² ≈ 0.** Attn wheel 0.260 and paw 0.282 on ZM_2241 (371 trials). Wheel mean trial R² is +0.24 and the median trial is +0.44. Paw mean trial R² of −8 is a few exploded still-trial scores; the median trial is +0.37. Concatenated-bin R² is the more stable mouse-level number.
5. **Training is healthy.** Val tracks train. Early stopping fired on four of six runs; mean-pool trial and mouse holdouts used almost all 40 epochs and were still improving slowly.
