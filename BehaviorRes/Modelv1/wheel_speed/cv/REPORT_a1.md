# Modelv1 — Control A1 (set encoder without anatomical metadata)

Built 2026-09-24T13:42:45.844076+00:00

## What this is

Control A1 from the neural-decoding framework (PDF §6.12.1): the same set encoder, pooling, causal GRU, and 1-d behavior head as Formulation A, but the per-unit encoder is φ_θ([log1p count, session mean log-rate]). CCF xyz and Allen-region embeddings are removed. This tests whether the direct set model learns a useful cross-recording representation from activity alone.

RightContra: ? sessions. Motor units/session n/a. Trials shorter than 13 bins after motor filtering are dropped; T is cropped at 128 bins (2.56 s). This folder trains a 1-d head on **wheel speed |ω|** only; sibling folders hold the other behaviors.

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
| mouse_lomo | attn | wheel_speed | 32 | 0.269 | 0.245 | 0.309 | 0.106 | 0.354 |
| mouse_lomo | mean | wheel_speed | 32 | 0.245 | 0.225 | 0.294 | 0.090 | 0.363 |
| session_loso | attn | wheel_speed | 52 | 0.282 | 0.227 | 0.312 | 0.122 | 0.387 |
| session_loso | mean | wheel_speed | 52 | 0.257 | 0.227 | 0.286 | 0.093 | 0.380 |
| trial_extrapolation | attn | wheel_speed | 50 | -0.080 | 0.850 | 0.023 | -0.422 | 0.218 |
| trial_extrapolation | mean | wheel_speed | 50 | -0.278 | 1.365 | 0.014 | -0.638 | 0.064 |
| trial_repeat | attn | wheel_speed | 5 | 0.293 | 0.044 | 0.286 | 0.193 | 0.445 |
| trial_repeat | mean | wheel_speed | 5 | 0.275 | 0.026 | 0.270 | 0.176 | 0.455 |

Per-fold scores are in `scores.csv` (column `fold_id`).

### Best pool per task × target (concatenated-bin R²)

- trial_repeat / wheel_speed: **attn** mean R²=0.293 (n=5 folds)
- trial_extrapolation / wheel_speed: **attn** mean R²=-0.080 (n=50 folds)
- session_loso / wheel_speed: **attn** mean R²=0.282 (n=52 folds)
- mouse_lomo / wheel_speed: **attn** mean R²=0.269 (n=32 folds)

## Training diagnostics

### trial_repeat


### trial_extrapolation


### session_loso


### mouse_lomo


Plots: `train_curves_a1.png`, `r2_concat_a1.png`, `r2_trial_a1.png`, `examples_<task>_a1.png`, and `a1_vs_full.png` when the full model scores exist.
