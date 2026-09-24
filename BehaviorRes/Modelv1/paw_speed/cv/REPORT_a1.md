# Modelv1 — Control A1 (set encoder without anatomical metadata)

Built 2026-09-24T13:42:59.577089+00:00

## What this is

Control A1 from the neural-decoding framework (PDF §6.12.1): the same set encoder, pooling, causal GRU, and 1-d behavior head as Formulation A, but the per-unit encoder is φ_θ([log1p count, session mean log-rate]). CCF xyz and Allen-region embeddings are removed. This tests whether the direct set model learns a useful cross-recording representation from activity alone.

RightContra: ? sessions. Motor units/session n/a. Trials shorter than 13 bins after motor filtering are dropped; T is cropped at 128 bins (2.56 s). This folder trains a 1-d head on **paw speed (2D, x-y)** only; sibling folders hold the other behaviors.

## Holdouts

Cross-validation protocol: the model is retrained from scratch on every fold. `trial_repeat` draws several random trial holdouts (same 10%-of-sessions / 10%-of-trials scheme as the original split, seeds 1000…). `trial_extrapolation` is one fold per session: train on the first 90% of that session's ordered trials, test on the last 10%. `session_loso` leaves one session out; `mouse_lomo` leaves one mouse out. The original single-split protocol is `--protocol fixed` (default).

- **mouse_lomo**: 32 fold(s) aggregated from jobs.
- **session_loso**: 52 fold(s) aggregated from jobs.
- **trial_extrapolation**: 50 fold(s) aggregated from jobs.
- **trial_repeat**: 5 fold(s) aggregated from jobs.

## Model

Unit MLP (log1p count, session mean log-rate (Control A1)): Linear(2→64), GELU, Linear(64→64). Mean pool applies Linear(64→64) to the masked average. Attention pool uses α ∝ exp(q⊤ tanh(W e)) over units. GRU hidden 64, decoder Dropout–Linear–GELU–Linear → 1. Train-time unit dropout 0.15.

Training loss is trial-balanced MSE on z-scored targets (PDF eq. 7): each trial contributes equally, then trials are averaged. AdamW lr=0.001, weight decay=0.0001, batch 32, max 40 epochs, patience 5.

## Results

Cross-validation summary (mean ± std across folds of concatenated-bin R²).

| task | pool | target | n folds | R² mean | R² std | R² median | mean trial R² | median trial R² |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mouse_lomo | attn | paw_speed | 32 | 0.134 | 0.240 | 0.169 | -29.470 | 0.165 |
| mouse_lomo | mean | paw_speed | 32 | 0.127 | 0.234 | 0.180 | -28.549 | 0.176 |
| session_loso | attn | paw_speed | 52 | 0.136 | 0.268 | 0.181 | -25.316 | 0.006 |
| session_loso | mean | paw_speed | 52 | 0.134 | 0.249 | 0.178 | -25.615 | 0.020 |
| trial_extrapolation | attn | paw_speed | 50 | -0.177 | 0.902 | 0.096 | -1.690 | 0.057 |
| trial_extrapolation | mean | paw_speed | 50 | -0.258 | 1.023 | 0.017 | -2.042 | 0.019 |
| trial_repeat | attn | paw_speed | 5 | 0.221 | 0.048 | 0.216 | -5.281 | 0.260 |
| trial_repeat | mean | paw_speed | 5 | 0.215 | 0.042 | 0.215 | -6.128 | 0.299 |

Per-fold scores are in `scores.csv` (column `fold_id`).

### Best pool per task × target (concatenated-bin R²)

- trial_repeat / paw_speed: **attn** mean R²=0.221 (n=5 folds)
- trial_extrapolation / paw_speed: **attn** mean R²=-0.177 (n=50 folds)
- session_loso / paw_speed: **attn** mean R²=0.136 (n=52 folds)
- mouse_lomo / paw_speed: **attn** mean R²=0.134 (n=32 folds)

## Training diagnostics

### trial_repeat


### trial_extrapolation


### session_loso


### mouse_lomo


Plots: `train_curves_a1.png`, `r2_concat_a1.png`, `r2_trial_a1.png`, `examples_<task>_a1.png`, and `a1_vs_full.png` when the full model scores exist.
