# Modelv1 — Formulation A set encoder on ModelDataRightContra

Built 2026-09-24T13:42:44.793152+00:00

## What this is

Formulation A from the neural-decoding framework: a shared per-unit encoder φ_θ([log1p count, CCF xyz, region embedding, session mean log-rate]), permutation-invariant pooling, then a causal GRU and a 1-d head for one behavior (wheel speed |ω|). Only MOp/MOs units are used. Column *n* is not a global neuron identity.

RightContra: ? sessions. Motor units/session n/a. Trials shorter than 13 bins after motor filtering are dropped; T is cropped at 128 bins (2.56 s). This folder trains a 1-d head on **wheel speed |ω|** only; sibling folders hold the other behaviors.

## Holdouts

Cross-validation protocol: the model is retrained from scratch on every fold. `trial_repeat` draws several random trial holdouts (same 10%-of-sessions / 10%-of-trials scheme as the original split, seeds 1000…). `trial_extrapolation` is one fold per session: train on the first 90% of that session's ordered trials, test on the last 10%. `session_loso` leaves one session out; `mouse_lomo` leaves one mouse out. The original single-split protocol is `--protocol fixed` (default).

- **mouse_lomo**: 32 fold(s) aggregated from jobs.
- **session_loso**: 52 fold(s) aggregated from jobs.
- **trial_extrapolation**: 50 fold(s) aggregated from jobs.
- **trial_repeat**: 5 fold(s) aggregated from jobs.

## Model

Unit MLP (log1p count, CCF xyz, region embedding, session mean log-rate): Linear(21→64), GELU, Linear(64→64). Mean pool applies Linear(64→64) to the masked average. Attention pool uses α ∝ exp(q⊤ tanh(W e)) over units. GRU hidden 64, decoder Dropout–Linear–GELU–Linear → 1. Train-time unit dropout 0.15.

Training loss is trial-balanced MSE on z-scored targets (PDF eq. 7): each trial contributes equally, then trials are averaged. AdamW lr=0.001, weight decay=0.0001, batch 32, max 40 epochs, patience 5.

## Results

Cross-validation summary (mean ± std across folds of concatenated-bin R²).

| task | pool | target | n folds | R² mean | R² std | R² median | mean trial R² | median trial R² |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mouse_lomo | attn | wheel_speed | 32 | 0.127 | 0.279 | 0.202 | -0.077 | 0.169 |
| mouse_lomo | mean | wheel_speed | 32 | 0.203 | 0.252 | 0.259 | 0.030 | 0.274 |
| session_loso | attn | wheel_speed | 52 | 0.180 | 0.291 | 0.223 | 0.004 | 0.266 |
| session_loso | mean | wheel_speed | 52 | 0.194 | 0.250 | 0.237 | 0.001 | 0.296 |
| trial_extrapolation | attn | wheel_speed | 50 | -0.298 | 1.448 | 0.014 | -0.654 | 0.066 |
| trial_extrapolation | mean | wheel_speed | 50 | -0.315 | 1.232 | -0.016 | -0.699 | 0.056 |
| trial_repeat | attn | wheel_speed | 5 | 0.313 | 0.065 | 0.285 | 0.189 | 0.489 |
| trial_repeat | mean | wheel_speed | 5 | 0.312 | 0.047 | 0.295 | 0.175 | 0.500 |

Per-fold scores are in `scores.csv` (column `fold_id`).

### Best pool per task × target (concatenated-bin R²)

- trial_repeat / wheel_speed: **attn** mean R²=0.313 (n=5 folds)
- trial_extrapolation / wheel_speed: **attn** mean R²=-0.298 (n=50 folds)
- session_loso / wheel_speed: **mean** mean R²=0.194 (n=52 folds)
- mouse_lomo / wheel_speed: **mean** mean R²=0.203 (n=32 folds)

## Training diagnostics

### trial_repeat


### trial_extrapolation


### session_loso


### mouse_lomo


Plots: `train_curves.png`, `r2_concat.png`, `r2_trial.png`, `r2_trial_median.png`, `r2_cv_folds_<target>.png`, `examples_<task>.png`.
