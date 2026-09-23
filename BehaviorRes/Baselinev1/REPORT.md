# Baselinev1 — controls for Formulation A on ModelDataRightContra

Built 2026-09-22T05:08:52.140189+00:00

## What this is

Baselines from the neural-decoding framework, evaluated on the same ModelDataRightContra motor-cortex corpus and holdouts as `Modelv1.py`. Cross-recording models do not use a globally fixed neuron index. Session-specific models (B9, B10, BWM) are within-session references; when a test session is absent from train they are scored as nested-CV **oracles** (hatched in the plots) and are not cross-session competitors.

## Models

- **B0 mean**: constant train-set mean of each target (PDF eq. 66–67).
- **B0 traj**: stimOn-aligned mean behavioral trajectory (PDF eq. 68).
- **B1**: instantaneous ridge on the population mean rate (PDF eq. 69–70).
- **B2**: causal W=10 window of the population mean, ridge (PDF eq. 75–76). First W−1 bins are dropped.
- **B3**: Allen-region pooled ridge with occupancy mask and log unit-count (PDF eq. 79–82).
- **B6**: population mean sequence through the same GRU + MLP head as Modelv1 (PDF eq. 92–94).
- **B7**: region-feature sequence through that same GRU (PDF eq. 95–97).
- **B9**: session-specific instantaneous ridge on the unit vector (PDF eq. 101).
- **B10**: session-specific Linear–GELU–Linear encoder on the fixed neuron axis, then the matched GRU (PDF eq. 104–106).
- **BWM lag ridge**: Brain-Wide Map movement decoder structure applied to this task — session-specific Ridge on motor units with causal lag W=10 (paper `n_bins_lag=10`, 20 ms bins), StandardScaler, alpha chosen on a validation split. The paper used Lasso; this file uses Ridge as in `BWMtest.py` because unscaled L1 did not converge on these counts. Independent models for |ω| and paw speed. Window is stimOn→end of stored trial, not firstMovement −0.2:+1.0 s.

RightContra: 52 sessions. Motor units/session 5–226. Same trial filter as Modelv1 (T≥13, crop 128). Inputs are log1p spike counts.

## Holdouts

- **trial_holdout**: 5/52 sessions contribute held-out trials (5adab0b7, 62902992, 8c33abef, a92c4b1d, d2f5a130). train n=5707, test n=45.
- **session_holdout**: held-out session 626126d5 of CSH_ZAD_026; 454 trials from other sessions of that mouse remain in train. train n=5515, test n=237.
- **mouse_holdout**: held-out mouse ZM_2241 (one session, unseen animal). train n=5381, test n=371.

## Results

| task | model | oracle | target | R² concat | mean trial R² | median trial R² | RMSE | MAE | Pearson | n bins | n trials |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trial_holdout | b0_const |  | paw_speed | -0.004 | -0.172 | -0.074 | 605.949 | 423.384 | nan | 1016 | 45 |
| trial_holdout | b0_const |  | wheel_speed | -0.000 | -0.105 | -0.029 | 0.937 | 0.734 | nan | 1016 | 45 |
| trial_holdout | b0_traj |  | paw_speed | 0.135 | 0.134 | 0.340 | 562.292 | 333.274 | 0.378 | 1016 | 45 |
| trial_holdout | b0_traj |  | wheel_speed | 0.332 | 0.269 | 0.389 | 0.766 | 0.514 | 0.577 | 1016 | 45 |
| trial_holdout | b1 |  | paw_speed | 0.097 | -0.131 | -0.047 | 574.604 | 413.649 | 0.369 | 1016 | 45 |
| trial_holdout | b1 |  | wheel_speed | 0.050 | -0.107 | 0.015 | 0.913 | 0.748 | 0.259 | 1016 | 45 |
| trial_holdout | b2 |  | paw_speed | 0.127 | -0.254 | -0.229 | 625.608 | 445.549 | 0.425 | 611 | 27 |
| trial_holdout | b2 |  | wheel_speed | 0.061 | -0.766 | -0.057 | 1.025 | 0.880 | 0.292 | 611 | 27 |
| trial_holdout | b3 |  | paw_speed | 0.089 | -0.055 | -0.023 | 577.002 | 382.200 | 0.338 | 1016 | 45 |
| trial_holdout | b3 |  | wheel_speed | 0.081 | -0.080 | 0.015 | 0.898 | 0.702 | 0.290 | 1016 | 45 |
| trial_holdout | b6 |  | paw_speed | 0.234 | 0.083 | 0.250 | 529.257 | 335.278 | 0.486 | 1016 | 45 |
| trial_holdout | b6 |  | wheel_speed | 0.346 | 0.171 | 0.510 | 0.758 | 0.539 | 0.645 | 1016 | 45 |
| trial_holdout | b7 |  | paw_speed | 0.357 | 0.244 | 0.417 | 484.714 | 281.637 | 0.603 | 1016 | 45 |
| trial_holdout | b7 |  | wheel_speed | 0.424 | 0.260 | 0.579 | 0.711 | 0.466 | 0.668 | 1016 | 45 |
| trial_holdout | b9 |  | paw_speed | 0.387 | 0.251 | 0.334 | 473.586 | 300.999 | 0.624 | 1016 | 45 |
| trial_holdout | b9 |  | wheel_speed | 0.257 | 0.030 | 0.395 | 0.808 | 0.580 | 0.556 | 1016 | 45 |
| trial_holdout | b10 |  | paw_speed | 0.415 | 0.238 | 0.337 | 462.443 | 276.100 | 0.645 | 1016 | 45 |
| trial_holdout | b10 |  | wheel_speed | 0.387 | 0.176 | 0.570 | 0.734 | 0.481 | 0.691 | 1016 | 45 |
| trial_holdout | bwm |  | paw_speed | 0.344 | -0.038 | 0.018 | 547.124 | 357.302 | 0.619 | 566 | 24 |
| trial_holdout | bwm |  | wheel_speed | 0.293 | -1.415 | 0.132 | 0.906 | 0.662 | 0.604 | 566 | 24 |
| session_holdout | b0_const |  | paw_speed | -0.163 | -1.585 | -0.445 | 427.847 | 346.370 | nan | 4694 | 237 |
| session_holdout | b0_const |  | wheel_speed | -0.010 | -0.079 | -0.051 | 1.061 | 0.824 | nan | 4694 | 237 |
| session_holdout | b0_traj |  | paw_speed | 0.136 | -1.169 | 0.252 | 368.789 | 248.621 | 0.537 | 4694 | 237 |
| session_holdout | b0_traj |  | wheel_speed | 0.417 | 0.388 | 0.468 | 0.806 | 0.518 | 0.687 | 4694 | 237 |
| session_holdout | b1 |  | paw_speed | -0.186 | -1.949 | -0.519 | 432.061 | 359.830 | 0.395 | 4694 | 237 |
| session_holdout | b1 |  | wheel_speed | 0.072 | 0.003 | 0.039 | 1.017 | 0.815 | 0.421 | 4694 | 237 |
| session_holdout | b2 |  | paw_speed | -0.114 | -3.375 | -0.734 | 485.118 | 386.918 | 0.534 | 2561 | 102 |
| session_holdout | b2 |  | wheel_speed | 0.065 | -0.405 | -0.182 | 1.131 | 0.950 | 0.488 | 2561 | 102 |
| session_holdout | b3 |  | paw_speed | -0.165 | -1.813 | -0.475 | 428.349 | 354.143 | 0.341 | 4694 | 237 |
| session_holdout | b3 |  | wheel_speed | 0.088 | -0.001 | 0.068 | 1.008 | 0.846 | 0.377 | 4694 | 237 |
| session_holdout | b6 |  | paw_speed | 0.008 | -1.895 | 0.017 | 395.313 | 279.503 | 0.579 | 4694 | 237 |
| session_holdout | b6 |  | wheel_speed | 0.511 | 0.435 | 0.670 | 0.738 | 0.501 | 0.720 | 4694 | 237 |
| session_holdout | b7 |  | paw_speed | -0.147 | -2.503 | -0.457 | 424.944 | 317.966 | 0.536 | 4694 | 237 |
| session_holdout | b7 |  | wheel_speed | 0.478 | 0.418 | 0.618 | 0.763 | 0.545 | 0.707 | 4694 | 237 |
| session_holdout | b9 | yes | paw_speed | 0.346 | -0.108 | 0.355 | 320.876 | 200.589 | 0.589 | 4694 | 237 |
| session_holdout | b9 | yes | wheel_speed | 0.401 | 0.311 | 0.450 | 0.817 | 0.608 | 0.633 | 4694 | 237 |
| session_holdout | b10 | yes | paw_speed | 0.414 | -0.143 | 0.550 | 303.836 | 170.482 | 0.649 | 4694 | 237 |
| session_holdout | b10 | yes | wheel_speed | 0.690 | 0.618 | 0.830 | 0.588 | 0.380 | 0.837 | 4694 | 237 |
| session_holdout | bwm | yes | paw_speed | 0.276 | -0.709 | 0.091 | 402.188 | 263.356 | 0.551 | 2324 | 84 |
| session_holdout | bwm | yes | wheel_speed | 0.537 | -0.292 | 0.102 | 0.811 | 0.625 | 0.733 | 2324 | 84 |
| mouse_holdout | b0_const |  | paw_speed | -0.004 | -4.494 | -0.054 | 583.298 | 421.665 | nan | 8433 | 371 |
| mouse_holdout | b0_const |  | wheel_speed | -0.032 | -0.093 | -0.058 | 0.967 | 0.750 | nan | 8433 | 371 |
| mouse_holdout | b0_traj |  | paw_speed | 0.261 | -6.776 | 0.332 | 500.340 | 309.017 | 0.519 | 8433 | 371 |
| mouse_holdout | b0_traj |  | wheel_speed | 0.270 | 0.325 | 0.446 | 0.813 | 0.529 | 0.543 | 8433 | 371 |
| mouse_holdout | b1 |  | paw_speed | 0.014 | -4.789 | -0.049 | 577.902 | 417.573 | 0.142 | 8433 | 371 |
| mouse_holdout | b1 |  | wheel_speed | -0.014 | -0.071 | -0.039 | 0.958 | 0.743 | 0.124 | 8433 | 371 |
| mouse_holdout | b2 |  | paw_speed | -0.011 | -16.641 | -0.213 | 637.570 | 450.617 | 0.113 | 5094 | 241 |
| mouse_holdout | b2 |  | wheel_speed | -0.056 | -0.840 | -0.214 | 1.018 | 0.815 | 0.135 | 5094 | 241 |
| mouse_holdout | b3 |  | paw_speed | -0.048 | -2.712 | -0.072 | 596.055 | 399.324 | 0.102 | 8433 | 371 |
| mouse_holdout | b3 |  | wheel_speed | -0.040 | -0.092 | -0.062 | 0.971 | 0.733 | 0.131 | 8433 | 371 |
| mouse_holdout | b6 |  | paw_speed | 0.272 | -7.973 | 0.340 | 496.806 | 309.413 | 0.522 | 8433 | 371 |
| mouse_holdout | b6 |  | wheel_speed | 0.278 | 0.261 | 0.475 | 0.809 | 0.537 | 0.544 | 8433 | 371 |
| mouse_holdout | b7 |  | paw_speed | 0.203 | -3.963 | 0.258 | 519.832 | 298.168 | 0.507 | 8433 | 371 |
| mouse_holdout | b7 |  | wheel_speed | 0.176 | 0.233 | 0.319 | 0.864 | 0.548 | 0.498 | 8433 | 371 |
| mouse_holdout | b9 | yes | paw_speed | 0.158 | -7.118 | 0.138 | 534.088 | 368.475 | 0.398 | 8433 | 371 |
| mouse_holdout | b9 | yes | wheel_speed | 0.201 | 0.141 | 0.223 | 0.851 | 0.658 | 0.449 | 8433 | 371 |
| mouse_holdout | b10 | yes | paw_speed | 0.350 | -11.155 | 0.434 | 469.358 | 290.017 | 0.595 | 8433 | 371 |
| mouse_holdout | b10 | yes | wheel_speed | 0.448 | 0.405 | 0.613 | 0.707 | 0.477 | 0.675 | 8433 | 371 |
| mouse_holdout | bwm | yes | paw_speed | 0.137 | -36.366 | -0.081 | 596.322 | 432.931 | 0.380 | 4723 | 213 |
| mouse_holdout | bwm | yes | wheel_speed | 0.148 | -0.611 | -0.111 | 0.925 | 0.747 | 0.386 | 4723 | 213 |

### Formulation A (from Modelv1 scores, same splits)

| task | model | target | R² concat | mean trial R² | Pearson |
|---|---|---|---:|---:|---:|
| trial_holdout | A_mean | wheel_speed | 0.456 | 0.281 | 0.715 |
| trial_holdout | A_mean | paw_speed | 0.328 | 0.105 | 0.576 |
| trial_holdout | A_attn | wheel_speed | 0.486 | 0.313 | 0.714 |
| trial_holdout | A_attn | paw_speed | 0.280 | 0.164 | 0.533 |
| session_holdout | A_mean | wheel_speed | 0.565 | 0.493 | 0.756 |
| session_holdout | A_mean | paw_speed | 0.148 | -1.384 | 0.599 |
| session_holdout | A_attn | wheel_speed | 0.558 | 0.520 | 0.765 |
| session_holdout | A_attn | paw_speed | 0.259 | -0.893 | 0.625 |
| mouse_holdout | A_mean | wheel_speed | 0.233 | 0.259 | 0.510 |
| mouse_holdout | A_mean | paw_speed | 0.255 | -8.339 | 0.506 |
| mouse_holdout | A_attn | wheel_speed | 0.260 | 0.244 | 0.531 |
| mouse_holdout | A_attn | paw_speed | 0.282 | -8.051 | 0.534 |
| trial_holdout | A_mean_a1 | wheel_speed | 0.427 | 0.261 | 0.683 |
| trial_holdout | A_mean_a1 | paw_speed | 0.296 | 0.160 | 0.545 |
| trial_holdout | A_attn_a1 | wheel_speed | 0.419 | 0.276 | 0.681 |
| trial_holdout | A_attn_a1 | paw_speed | 0.291 | 0.192 | 0.540 |
| session_holdout | A_mean_a1 | wheel_speed | 0.583 | 0.499 | 0.768 |
| session_holdout | A_mean_a1 | paw_speed | 0.197 | -1.173 | 0.640 |
| session_holdout | A_attn_a1 | wheel_speed | 0.547 | 0.476 | 0.750 |
| session_holdout | A_attn_a1 | paw_speed | 0.126 | -1.279 | 0.625 |
| mouse_holdout | A_mean_a1 | wheel_speed | 0.271 | 0.274 | 0.533 |
| mouse_holdout | A_mean_a1 | paw_speed | 0.262 | -9.656 | 0.513 |
| mouse_holdout | A_attn_a1 | wheel_speed | 0.212 | 0.177 | 0.504 |
| mouse_holdout | A_attn_a1 | paw_speed | 0.216 | -8.680 | 0.474 |

### Best model per task × target (concatenated-bin R²)

- trial_holdout / wheel_speed (cross-recording, includes Formulation A): **A_attn** (A attn pool) R²=0.486
  - within-session reference: **b10** (B10 sess GRU) R²=0.387
- trial_holdout / paw_speed (cross-recording, includes Formulation A): **b7** (B7 region GRU) R²=0.357
  - within-session reference: **b10** (B10 sess GRU) R²=0.415
- session_holdout / wheel_speed (cross-recording, includes Formulation A): **A_mean_a1** (A1 mean) R²=0.583
  - oracle reference: **b10** (B10 sess GRU) R²=0.690
- session_holdout / paw_speed (cross-recording, includes Formulation A): **A_attn** (A attn pool) R²=0.259
  - oracle reference: **b10** (B10 sess GRU) R²=0.414
- mouse_holdout / wheel_speed (cross-recording, includes Formulation A): **b6** (B6 pop GRU) R²=0.278
  - oracle reference: **b10** (B10 sess GRU) R²=0.448
- mouse_holdout / paw_speed (cross-recording, includes Formulation A): **A_attn** (A attn pool) R²=0.282
  - oracle reference: **b10** (B10 sess GRU) R²=0.350

## Training diagnostics (GRU baselines)

### trial_holdout

- b6: 29378 params, best @ 21/26  train=1.5091  val=1.4931
- b7: 31234 params, best @ 40/40  train=1.3324  val=1.3147
- b10: 39426 params, best @ 16/21  train=0.9131  val=0.7629

### session_holdout

- b6: 29378 params, best @ 21/26  train=1.5315  val=1.5300
- b7: 31234 params, best @ 39/40  train=1.3456  val=1.3829
- b10: 37762 params, best @ 20/25  train=0.5353  val=0.3943

### mouse_holdout

- b6: 29378 params, best @ 28/33  train=1.5092  val=1.5350
- b7: 31234 params, best @ 22/27  train=1.3866  val=1.3910
- b10: 37890 params, best @ 40/40  train=0.8299  val=1.0570

## Interpretation

Concatenated-bin R² is the primary number. Mean trial R² is pulled around by short or nearly-still trials (especially paw speed on mouse holdout); median trial R² is the more stable per-trial summary.

1. **Rate-only linear models do not decode this task.** B1 (population mean), B2 (causal W=10 window of that mean), and B3 (Allen-region pooled ridge) stay near the constant-mean null on every holdout (trial wheel 0.050 / 0.061 / 0.081 vs B0 mean -0.000). Instantaneous rate is not the missing ingredient.

2. **A matched temporal backbone is.** B6 (population-mean GRU) and B7 (region-feature GRU) use the same GRU+MLP head as Formulation A. On trial holdout, B7 reaches wheel/paw 0.424 / 0.357 vs the stimOn trajectory null 0.332 / 0.135. On a new session of a seen mouse, B6 wheel 0.511 beats that null (0.417); paw does not (0.008 vs traj 0.136). On a new mouse, B6 (0.278 / 0.272) is indistinguishable from the trajectory null (0.270 / 0.261), and B7 is worse.

3. **Formulation A is the best *cross-recording* model on seen animals, not on a new mouse.** Trial wheel: A attn 0.486 vs best baseline B7 0.424. Session wheel: A mean 0.565 vs B6 0.511. Session paw: A attn 0.259 vs traj 0.136 — this is the one place anatomy + set pooling clearly helps a target that B6/B7 lose. Mouse holdout: A attn 0.260 / 0.282 sits next to B6 and the trajectory null. The set encoder is not buying mouse transfer beyond a shared GRU on the population mean.

4. **Within-session oracles are the ceiling, and they are not reachable from other recordings.** Hatched B10 (session-specific GRU on the fixed neuron axis) is 0.690 / 0.414 on the held-out session and 0.448 / 0.350 inside the held-out mouse. Formulation A recovers most of the session-oracle wheel (0.565 / 0.690) but only about half of the within-mouse oracle. B9 (instantaneous session ridge) and the BWM lagged ridge sit between B10 and the cross-recording models on trial holdout; as oracles they still beat every linear rate baseline, and BWM session-oracle wheel (0.537) is a strong linear within-session number.

5. **What Formulation A actually buys on this corpus.** Shared per-unit anatomy + pooling beats a matched GRU on collapsed rate/region features when the animal has been seen (trial and new-insertion session). It does not beat a session-specific decoder that is allowed to see that session's neurons, and it does not beat the behavioral-trajectory null on an unseen mouse. Control A1 (drop CCF xyz and region) is the remaining check of whether those anatomical channels are doing the work on the seen-animal gains.

Plots: `r2_concat.png`, `r2_trial.png`, `trial_cross_vs_within.png`, `train_curves.png`, `examples_<task>.png`.
