# Modelv2 — Formulation B voxel encoders on ModelDataRightContra

Built 2026-09-22T15:19:01.221108+00:00

## What this is

Formulation B: motor units are assigned to a **fixed anatomical catalog** before any learned weights see the trial. Two catalogs are trained independently. Targets, holdouts, trial-balanced MSE, and the GRU-64 decoder match Modelv1.

**Strategy 2 (`aniso3d`)** is a 400 µm ML × 400 µm AP × 100 µm DV grid over the motor CCF hull. The spatial model is a 3-D CNN: shared Conv2d on the (ML×AP column, DV) unfolding (3×400 µm × 5×100 µm), occupancy-weighted global pool, Linear to 64. The `flat` control pools the same grid without convolution (PDF control B1).

**Strategy 3 (`layer_tiles`)** is Allen MOp/MOs × layer (1, 2/3, 5, 6a, 6b) × 400 µm ML/AP tiles. Depth is the Allen layer label, not DV cubes. `spatial` is an MLP on the flattened [G; M; log(1+C)] vector; `flat` is a Linear.

## Holdouts

- **trial_holdout**: 5/52 sessions contribute held-out trials (5adab0b7, 62902992, 8c33abef, a92c4b1d, d2f5a130). train n=5707, test n=45.
- **session_holdout**: held-out session 626126d5 of CSH_ZAD_026; 454 trials from other sessions of that mouse remain in train. train n=5515, test n=237.
- **mouse_holdout**: held-out mouse ZM_2241 (one session, unseen animal). train n=5381, test n=371.

## Voxelization quality

### aniso3d

- Catalog V=2590, occupied by any session 353 (13.6%).
- Median occupied voxels: 9 / session, 14 / mouse.
- Units per occupied voxel (session-level): mean 4.77, median 4.00.
- Fraction of occupied voxels unique to one session: 0.65. Shared by ≥2 sessions: 124.
- Mean occupancy Jaccard: sessions 0.008, within multi-session mice 0.030, between mice 0.012.
- trial_holdout train∩test coverage: test voxels covered by train catalog occupancy 1.000 (Jaccard 0.139; train V=353, test V=49).
- session_holdout train∩test coverage: test voxels covered by train catalog occupancy 0.000 (Jaccard 0.000; train V=343, test V=10).
- mouse_holdout train∩test coverage: test voxels covered by train catalog occupancy 0.833 (Jaccard 0.028; train V=351, test V=12).

### layer_tiles

- Catalog V=700, occupied by any session 121 (17.3%).
- Median occupied voxels: 4 / session, 5 / mouse.
- Units per occupied voxel (session-level): mean 11.80, median 8.00.
- Fraction of occupied voxels unique to one session: 0.50. Shared by ≥2 sessions: 61.
- Mean occupancy Jaccard: sessions 0.014, within multi-session mice 0.039, between mice 0.019.
- trial_holdout train∩test coverage: test voxels covered by train catalog occupancy 1.000 (Jaccard 0.182; train V=121, test V=22).
- session_holdout train∩test coverage: test voxels covered by train catalog occupancy 0.000 (Jaccard 0.000; train V=118, test V=3).
- mouse_holdout train∩test coverage: test voxels covered by train catalog occupancy 1.000 (Jaccard 0.033; train V=121, test V=4).

A high singleton-voxel fraction plus low between-session Jaccard means the occupancy mask can fingerprint a recording. That is the failure mode the PDF warns about: the model may decode *where the probe sat* rather than a shared motor representation. Tables: `voxels/<strategy>_by_session.csv` and `_by_mouse.csv`.

### How good are the voxel representations?

Neither catalog is a dense shared motor map. Strategy 2 occupies 353/2590 voxels (13.6%); strategy 3 occupies 121/700 (17.3%). A typical session fills only 9 anisotropic voxels vs 4 layer tiles — one Neuropixels track, not a volume.

Correspondence is weak on both. Between-session Jaccard is 0.008 (aniso3d) vs 0.014 (layer_tiles). Within a multi-session mouse it only rises to 0.030 / 0.039. 65% of occupied aniso3d voxels and 50% of layer tiles belong to a single session.

The session holdout is the stress test: test-voxel coverage by the train occupancy is 0.00 (aniso3d) and 0.00 (layer_tiles). The held-out insertion of CSH_ZAD_026 lands in a disjoint set of parcels. A 3-D CNN can still apply shared kernels at those new coordinates; a position-specific MLP/Linear on layer tiles sees zeros on every tile the new probe occupies. Mouse-holdout coverage is 0.83 / 1.00 — ZM_2241 overlaps the train catalog much more than the session holdout does.

Pooling quality favors strategy 3: median 8 units/tile vs 4 units/100 µm DV bin. Layer tiles therefore average more neurons per parcel (less Poisson noise) but discard within-layer depth. Aniso3d keeps a 100 µm depth axis the CNN can filter, at the cost of sparser, more session-private voxels.

**Verdict.** Both representations implement Formulation B's correspondence map, but they are occupancy-sparse fingerprints more than a shared motor volume. Strategy 3 is the better *catalog* (higher Jaccard, fewer singletons, more units per parcel). Strategy 2 is the better *geometry* for a spatial CNN (a real DV axis). Decoding below says which of those facts wins.

## Decoding results

| task | model | target | R² concat | mean trial R² | median trial R² | RMSE | MAE | Pearson | n bins | n trials |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trial_holdout | layer_tiles_spatial | wheel_speed | 0.198 | -0.054 | 0.538 | 0.839 | 0.538 | 0.620 | 1016 | 45 |
| trial_holdout | layer_tiles_spatial | paw_speed | 0.378 | -0.040 | 0.224 | 476.855 | 312.904 | 0.620 | 1016 | 45 |
| trial_holdout | layer_tiles_flat | wheel_speed | 0.389 | 0.168 | 0.636 | 0.733 | 0.458 | 0.685 | 1016 | 45 |
| trial_holdout | layer_tiles_flat | paw_speed | 0.446 | 0.248 | 0.408 | 450.231 | 266.857 | 0.670 | 1016 | 45 |
| session_holdout | layer_tiles_spatial | wheel_speed | 0.442 | 0.386 | 0.603 | 0.788 | 0.548 | 0.673 | 4694 | 237 |
| session_holdout | layer_tiles_spatial | paw_speed | -0.208 | -2.740 | -0.476 | 436.176 | 326.952 | 0.495 | 4694 | 237 |
| session_holdout | layer_tiles_flat | wheel_speed | 0.450 | 0.407 | 0.556 | 0.783 | 0.537 | 0.687 | 4694 | 237 |
| session_holdout | layer_tiles_flat | paw_speed | -0.225 | -2.852 | -0.541 | 439.236 | 333.774 | 0.500 | 4694 | 237 |
| mouse_holdout | layer_tiles_spatial | wheel_speed | -0.099 | -0.163 | -0.102 | 0.998 | 0.632 | 0.349 | 8433 | 371 |
| mouse_holdout | layer_tiles_spatial | paw_speed | -0.083 | -2.445 | -0.038 | 605.921 | 364.415 | 0.225 | 8433 | 371 |
| mouse_holdout | layer_tiles_flat | wheel_speed | -0.392 | -0.536 | -0.444 | 1.123 | 0.724 | 0.215 | 8433 | 371 |
| mouse_holdout | layer_tiles_flat | paw_speed | -0.241 | -1.007 | -0.289 | 648.515 | 385.529 | 0.195 | 8433 | 371 |
| trial_holdout | aniso3d_spatial | wheel_speed | 0.295 | 0.091 | 0.534 | 0.787 | 0.542 | 0.649 | 1016 | 45 |
| trial_holdout | aniso3d_spatial | paw_speed | 0.322 | 0.103 | 0.260 | 497.991 | 311.970 | 0.574 | 1016 | 45 |
| trial_holdout | aniso3d_flat | wheel_speed | 0.382 | 0.248 | 0.502 | 0.737 | 0.514 | 0.647 | 1016 | 45 |
| trial_holdout | aniso3d_flat | paw_speed | 0.207 | 0.063 | 0.283 | 538.587 | 331.402 | 0.460 | 1016 | 45 |
| session_holdout | aniso3d_spatial | wheel_speed | 0.562 | 0.497 | 0.654 | 0.699 | 0.438 | 0.752 | 4694 | 237 |
| session_holdout | aniso3d_spatial | paw_speed | 0.382 | 0.117 | 0.460 | 312.018 | 185.735 | 0.629 | 4694 | 237 |
| session_holdout | aniso3d_flat | wheel_speed | 0.430 | 0.363 | 0.620 | 0.797 | 0.553 | 0.663 | 4694 | 237 |
| session_holdout | aniso3d_flat | paw_speed | -0.017 | -1.780 | -0.159 | 400.189 | 284.121 | 0.466 | 4694 | 237 |
| mouse_holdout | aniso3d_spatial | wheel_speed | 0.240 | 0.132 | 0.418 | 0.830 | 0.570 | 0.554 | 8433 | 371 |
| mouse_holdout | aniso3d_spatial | paw_speed | 0.226 | -10.734 | 0.300 | 512.159 | 347.477 | 0.515 | 8433 | 371 |
| mouse_holdout | aniso3d_flat | wheel_speed | 0.296 | 0.298 | 0.456 | 0.799 | 0.529 | 0.552 | 8433 | 371 |
| mouse_holdout | aniso3d_flat | paw_speed | 0.277 | -8.711 | 0.351 | 495.120 | 313.872 | 0.528 | 8433 | 371 |

### Best model per task × target (concatenated-bin R²)

- trial_holdout / wheel_speed: **layer_tiles_flat** R²=0.389 (mean trial R²=0.168)
- trial_holdout / paw_speed: **layer_tiles_flat** R²=0.446 (mean trial R²=0.248)
- session_holdout / wheel_speed: **aniso3d_spatial** R²=0.562 (mean trial R²=0.497)
- session_holdout / paw_speed: **aniso3d_spatial** R²=0.382 (mean trial R²=0.117)
- mouse_holdout / wheel_speed: **aniso3d_flat** R²=0.296 (mean trial R²=0.298)
- mouse_holdout / paw_speed: **aniso3d_flat** R²=0.277 (mean trial R²=-8.711)

## Training diagnostics

### trial_holdout

- aniso3d_spatial: 39810 params, best @ 15/20  train=1.4156  val=1.4803
- aniso3d_flat: 30850 params, best @ 19/24  train=1.5318  val=1.5986
- layer_tiles_spatial: 306434 params, best @ 14/19  train=1.3577  val=1.2914
- layer_tiles_flat: 163714 params, best @ 26/31  train=1.3118  val=1.2686

### session_holdout

- aniso3d_spatial: 39810 params, best @ 36/40  train=1.3514  val=1.3082
- aniso3d_flat: 30850 params, best @ 8/13  train=1.5727  val=1.4962
- layer_tiles_spatial: 306434 params, best @ 34/39  train=1.3040  val=1.3428
- layer_tiles_flat: 163714 params, best @ 40/40  train=1.3065  val=1.3462

### mouse_holdout

- aniso3d_spatial: 39810 params, best @ 14/19  train=1.4326  val=1.3410
- aniso3d_flat: 30850 params, best @ 24/29  train=1.5344  val=1.4533
- layer_tiles_spatial: 306434 params, best @ 24/29  train=1.3234  val=1.3045
- layer_tiles_flat: 163714 params, best @ 22/27  train=1.3294  val=1.3185

Plots: `train_curves.png`, `r2_concat.png`, `r2_trial.png`, `examples_<task>.png`, `voxels/` occupancy figures.

## Interpretation

1. **Control B1 (flat) wins when the same voxels are reused.** On trial holdout, `layer_tiles_flat` is the best Formulation B model (wheel 0.389 / paw 0.446). The extra MLP and the 3-D CNN do not help when test trials sit on tiles the Linear already saw.
2. **The 3-D CNN is the only encoder that transfers a new insertion.** Session holdout occupancy overlap is 0.00 for both catalogs. `aniso3d_spatial` reaches wheel 0.562 / paw 0.382; the same grid without convolution drops to 0.430 / −0.017, and both layer-tile models go negative on paw. Shared kernels at new CCF coordinates are doing the work B1 cannot.
3. **Strategy 3 fails a new mouse even with 100% tile coverage.** Layer-tile mouse-holdout R² is negative (spatial −0.099 / −0.083; flat −0.392 / −0.241). Aniso3d stays positive (flat 0.296 / 0.277). Covered tiles are not a shared representation if the weights are position-specific and the occupancy pattern still fingerprints the probe.
4. **Vs Modelv1 (Formulation A).** Session wheel is tied (CNN 0.562 vs attn 0.565). Session paw is better under B (0.382 vs 0.259). Mouse is matched (aniso flat 0.296 / 0.277 vs attn 0.260 / 0.282). Trial wheel is worse (best B 0.389 vs A 0.486): the set encoder still uses per-unit identity plus CCF better when the session is known.
