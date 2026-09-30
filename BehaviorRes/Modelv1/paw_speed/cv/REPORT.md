# Modelv1 — Formulation A set encoder on ModelDataRightContra

Built 2026-09-30T21:46:43.560777+00:00

## What this is

Formulation A from the neural-decoding framework: a shared per-unit encoder φ_θ([log1p count, CCF xyz, region embedding, session mean log-rate]), permutation-invariant pooling, then a causal GRU and a 1-d head for one behavior (wheel-paw speed (2D, x-y)). Only MOp/MOs units are used. Column *n* is not a global neuron identity.

RightContra: ? sessions. Motor units/session n/a. Trials shorter than 13 bins after motor filtering are dropped; T is cropped at 128 bins (2.56 s). Paw vx/vy/vz/speed are the right paw that turns the wheel (`used_paw` == `wheel_paws`). This folder trains a 1-d head on **wheel-paw speed (2D, x-y)** only; sibling folders hold the other behaviors.

## Holdouts

Cross-validation protocol: the model is retrained from scratch on every fold. `trial_repeat` uses every trial: for each trial, 10% of finite time bins are held out at random for interpolation (seeds 1000…); the leftover 90% of bins are training targets. `trial_extrapolation` is one fold on the same trials: the first 90% of each trial's finite bins train, the last 10% are held out so the model must extrapolate those late bins. `session_loso` leaves one session out; `mouse_lomo` leaves one mouse out. The original single-split protocol is `--protocol fixed` (default).

- **mouse_lomo**: 32 fold(s) aggregated from jobs.
- **session_loso**: 52 fold(s) aggregated from jobs.
- **trial_extrapolation**: 51 fold(s) aggregated from jobs.
- **trial_repeat**: 10 fold(s) aggregated from jobs.

## Model

Unit MLP (log1p count, CCF xyz, region embedding, session mean log-rate): Linear(21→64), GELU, Linear(64→64). Mean pool applies Linear(64→64) to the masked average. Attention pool uses α ∝ exp(q⊤ tanh(W e)) over units. GRU hidden 64, decoder Dropout–Linear–GELU–Linear → 1. Train-time unit dropout 0.15.

Training loss is trial-balanced MSE on z-scored targets (PDF eq. 7): each trial contributes equally, then trials are averaged. AdamW lr=0.001, weight decay=0.0001, batch 32, max 40 epochs, patience 5.

## Results

Cross-validation summary (mean ± std across folds of concatenated-bin R²).

| task | pool | target | n folds | R² mean | R² std | R² median | mean trial R² | median trial R² |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mouse_lomo | attn | paw_speed | 64 | 0.033 | 0.230 | 0.086 | -23.799 | 0.011 |
| mouse_lomo | mean | paw_speed | 64 | 0.028 | 0.232 | 0.081 | -19.018 | 0.012 |
| session_loso | attn | paw_speed | 104 | 0.033 | 0.254 | 0.097 | -20.685 | -0.172 |
| session_loso | mean | paw_speed | 104 | 0.051 | 0.334 | 0.083 | -28.744 | -0.379 |
| trial_extrapolation | attn | paw_speed | 51 | -0.296 | 1.133 | 0.023 | -75110.835 | -0.060 |
| trial_extrapolation | mean | paw_speed | 51 | -0.253 | 0.943 | 0.000 | -62014.968 | -0.057 |
| trial_repeat | attn | paw_speed | 10 | 0.240 | 0.047 | 0.235 | -41062.901 | -0.326 |
| trial_repeat | mean | paw_speed | 10 | 0.260 | 0.049 | 0.261 | -40065.706 | -0.262 |

Per-fold scores are in `scores.csv` (column `fold_id`).

### Best pool per task × target (concatenated-bin R²)

- trial_repeat / paw_speed: **mean** mean R²=0.260 (n=10 folds)
- trial_extrapolation / paw_speed: **mean** mean R²=-0.253 (n=51 folds)
- session_loso / paw_speed: **mean** mean R²=0.051 (n=104 folds)
- mouse_lomo / paw_speed: **attn** mean R²=0.033 (n=64 folds)

## Training diagnostics

### trial_repeat


### trial_extrapolation


### session_loso


### mouse_lomo


Plots: `train_curves.png`, `r2_concat.png`, `r2_trial.png`, `r2_trial_median.png`, `r2_cv_folds_<target>.png`, `examples_<task>.png`.
