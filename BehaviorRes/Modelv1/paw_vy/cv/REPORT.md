# Modelv1 — Formulation A set encoder on ModelDataRightContra

Built 2026-09-24T13:42:52.133828+00:00

## What this is

Formulation A from the neural-decoding framework: a shared per-unit encoder φ_θ([log1p count, CCF xyz, region embedding, session mean log-rate]), permutation-invariant pooling, then a causal GRU and a 1-d head for one behavior (paw vy). Only MOp/MOs units are used. Column *n* is not a global neuron identity.

RightContra: ? sessions. Motor units/session n/a. Trials shorter than 13 bins after motor filtering are dropped; T is cropped at 128 bins (2.56 s). This folder trains a 1-d head on **paw vy** only; sibling folders hold the other behaviors.

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
| mouse_lomo | attn | paw_vy | 32 | 0.004 | 0.041 | 0.006 | -0.622 | -0.085 |
| mouse_lomo | mean | paw_vy | 32 | -0.019 | 0.079 | -0.002 | -1.603 | -0.103 |
| session_loso | attn | paw_vy | 52 | 0.003 | 0.063 | 0.006 | -1.088 | -0.094 |
| session_loso | mean | paw_vy | 52 | -0.020 | 0.107 | -0.001 | -1.955 | -0.119 |
| trial_extrapolation | attn | paw_vy | 50 | -0.077 | 0.188 | -0.018 | -1.092 | -0.140 |
| trial_extrapolation | mean | paw_vy | 50 | -0.070 | 0.169 | -0.009 | -0.979 | -0.122 |
| trial_repeat | attn | paw_vy | 5 | 0.053 | 0.048 | 0.043 | -0.630 | -0.036 |
| trial_repeat | mean | paw_vy | 5 | 0.031 | 0.030 | 0.032 | -0.607 | -0.041 |

Per-fold scores are in `scores.csv` (column `fold_id`).

### Best pool per task × target (concatenated-bin R²)

- trial_repeat / paw_vy: **attn** mean R²=0.053 (n=5 folds)
- trial_extrapolation / paw_vy: **mean** mean R²=-0.070 (n=50 folds)
- session_loso / paw_vy: **attn** mean R²=0.003 (n=52 folds)
- mouse_lomo / paw_vy: **attn** mean R²=0.004 (n=32 folds)

## Training diagnostics

### trial_repeat


### trial_extrapolation


### session_loso


### mouse_lomo


Plots: `train_curves.png`, `r2_concat.png`, `r2_trial.png`, `r2_trial_median.png`, `r2_cv_folds_<target>.png`, `examples_<task>.png`.
