# Modelv1 — Control A1 (set encoder without anatomical metadata)

Built 2026-09-22T04:54:05.900476+00:00

## What this is

Control A1 from the neural-decoding framework (PDF §6.12.1): the same set encoder, pooling, causal GRU, and 2-d speed head as Formulation A, but the per-unit encoder is φ_θ([log1p count, session mean log-rate]). CCF xyz and Allen-region embeddings are removed. This tests whether the direct set model learns a useful cross-recording representation from activity alone.

RightContra: 52 sessions. Motor units/session 5–226. Trials shorter than 13 bins after motor filtering are dropped; T is cropped at 128 bins (2.56 s). Targets are |ω| (rad/s) and Lightning Pose 2D paw speed (DLC fallback).

## Holdouts

- **trial_holdout**: 5/52 sessions contribute held-out trials (5adab0b7, 62902992, 8c33abef, a92c4b1d, d2f5a130). train n=5707, test n=45.
- **session_holdout**: held-out session 626126d5 of CSH_ZAD_026; 454 trials from other sessions of that mouse remain in train. train n=5515, test n=237.
- **mouse_holdout**: held-out mouse ZM_2241 (one session, unseen animal). train n=5381, test n=371.

## Model

Unit MLP (log1p count, session mean log-rate (Control A1)): Linear(2→64), GELU, Linear(64→64). Mean pool applies Linear(64→64) to the masked average. Attention pool uses α ∝ exp(q⊤ tanh(W e)) over units. GRU hidden 64, decoder Dropout–Linear–GELU–Linear → 2. Train-time unit dropout 0.15.

Training loss is trial-balanced MSE on z-scored targets (PDF eq. 7): each trial contributes equally, then trials are averaged. AdamW lr=0.001, weight decay=0.0001, batch 32, max 40 epochs, patience 5.

## Results

| task | pool | target | R² concat | mean trial R² | median trial R² | RMSE | MAE | Pearson | n bins | n trials |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trial_holdout | mean | wheel_speed | 0.427 | 0.261 | 0.638 | 0.710 | 0.485 | 0.683 | 1016 | 45 |
| trial_holdout | mean | paw_speed | 0.296 | 0.160 | 0.338 | 507.515 | 310.183 | 0.545 | 1016 | 45 |
| trial_holdout | attn | wheel_speed | 0.419 | 0.276 | 0.507 | 0.714 | 0.492 | 0.681 | 1016 | 45 |
| trial_holdout | attn | paw_speed | 0.291 | 0.192 | 0.297 | 509.118 | 303.361 | 0.540 | 1016 | 45 |
| session_holdout | mean | wheel_speed | 0.583 | 0.499 | 0.719 | 0.682 | 0.447 | 0.768 | 4694 | 237 |
| session_holdout | mean | paw_speed | 0.197 | -1.173 | 0.082 | 355.579 | 236.928 | 0.640 | 4694 | 237 |
| session_holdout | attn | wheel_speed | 0.547 | 0.476 | 0.680 | 0.710 | 0.484 | 0.750 | 4694 | 237 |
| session_holdout | attn | paw_speed | 0.126 | -1.279 | 0.068 | 370.977 | 259.738 | 0.625 | 4694 | 237 |
| mouse_holdout | mean | wheel_speed | 0.271 | 0.274 | 0.448 | 0.813 | 0.540 | 0.533 | 8433 | 371 |
| mouse_holdout | mean | paw_speed | 0.262 | -9.656 | 0.330 | 500.225 | 321.243 | 0.513 | 8433 | 371 |
| mouse_holdout | attn | wheel_speed | 0.212 | 0.177 | 0.302 | 0.845 | 0.554 | 0.504 | 8433 | 371 |
| mouse_holdout | attn | paw_speed | 0.216 | -8.680 | 0.268 | 515.327 | 324.996 | 0.474 | 8433 | 371 |

### Best pool per task × target (concatenated-bin R²)

- trial_holdout / wheel_speed: **mean** R²=0.427 (mean trial R²=0.261)
- trial_holdout / paw_speed: **mean** R²=0.296 (mean trial R²=0.160)
- session_holdout / wheel_speed: **mean** R²=0.583 (mean trial R²=0.499)
- session_holdout / paw_speed: **mean** R²=0.197 (mean trial R²=-1.173)
- mouse_holdout / wheel_speed: **mean** R²=0.271 (mean trial R²=0.274)
- mouse_holdout / paw_speed: **mean** R²=0.262 (mean trial R²=-9.656)

## Training diagnostics

### trial_holdout

- mean: 37762 params, best @ 38/40  train=1.4540  val=1.4229
- attn: 37826 params, best @ 28/33  train=1.4453  val=1.3883

### session_holdout

- mean: 37762 params, best @ 21/26  train=1.5047  val=1.4934
- attn: 37826 params, best @ 20/25  train=1.4827  val=1.4694

### mouse_holdout

- mean: 37762 params, best @ 16/21  train=1.4913  val=1.5057
- attn: 37826 params, best @ 13/18  train=1.4650  val=1.4671

Plots: `train_curves_a1.png`, `r2_concat_a1.png`, `r2_trial_a1.png`, `examples_<task>_a1.png`, and `a1_vs_full.png` when the full model scores exist.
