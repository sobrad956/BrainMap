# Modelv1 — Formulation A set encoder on ModelDataRightContra

Built 2026-09-24T13:42:48.321784+00:00

## What this is

Formulation A from the neural-decoding framework: a shared per-unit encoder φ_θ([log1p count, CCF xyz, region embedding, session mean log-rate]), permutation-invariant pooling, then a causal GRU and a 1-d head for one behavior (paw vx). Only MOp/MOs units are used. Column *n* is not a global neuron identity.

RightContra: ? sessions. Motor units/session n/a. Trials shorter than 13 bins after motor filtering are dropped; T is cropped at 128 bins (2.56 s). This folder trains a 1-d head on **paw vx** only; sibling folders hold the other behaviors.

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
| mouse_lomo | attn | paw_vx | 32 | -0.101 | 0.215 | -0.023 | -2.677 | -0.348 |
| mouse_lomo | mean | paw_vx | 32 | -0.088 | 0.213 | -0.027 | -1.574 | -0.308 |
| session_loso | attn | paw_vx | 52 | -0.081 | 0.175 | -0.026 | -2.355 | -0.313 |
| session_loso | mean | paw_vx | 51 | -0.122 | 0.307 | -0.021 | -1.984 | -0.360 |
| trial_extrapolation | attn | paw_vx | 50 | -0.215 | 0.810 | -0.028 | -0.803 | -0.225 |
| trial_extrapolation | mean | paw_vx | 50 | -0.167 | 0.537 | -0.025 | -0.818 | -0.203 |
| trial_repeat | attn | paw_vx | 5 | 0.051 | 0.045 | 0.042 | -2.061 | -0.121 |
| trial_repeat | mean | paw_vx | 5 | 0.048 | 0.042 | 0.060 | -1.794 | -0.119 |

Per-fold scores are in `scores.csv` (column `fold_id`).

### Best pool per task × target (concatenated-bin R²)

- trial_repeat / paw_vx: **attn** mean R²=0.051 (n=5 folds)
- trial_extrapolation / paw_vx: **mean** mean R²=-0.167 (n=50 folds)
- session_loso / paw_vx: **attn** mean R²=-0.081 (n=52 folds)
- mouse_lomo / paw_vx: **mean** mean R²=-0.088 (n=32 folds)

## Training diagnostics

### trial_repeat


### trial_extrapolation


### session_loso


### mouse_lomo


Plots: `train_curves.png`, `r2_concat.png`, `r2_trial.png`, `r2_trial_median.png`, `r2_cv_folds_<target>.png`, `examples_<task>.png`.
