# MLtestcontra — motor-cortex decoding on ModelDataRightContra

Built 2026-09-18T12:54:39.950682+00:00

## Dataset and the neuron-identity problem

ModelDataRightContra is right-paw / right-hemisphere BWM trials. Only MOp/MOs units enter the models. Sessions still differ in how many motor units they have (5–226) and those units are *different cells* — column 0 of `spike_counts` in session A is not the same neuron as column 0 in session B. Concatenating raw (T, N) tensors across sessions would pretend otherwise. Every model here is permutation-invariant over the neuron axis: a shared encoder runs independently on each unit, then a masked mean / max / attention pool builds a population token per time bin.

Trials shorter than 13 bins after motor filtering are dropped. Sequences longer than 128 bins (2.56 s) are cropped from stimOn; that truncates <1% of raw RightContra trials. Spike counts are `log1p`. Wheel target is `|ω|` (rad/s). Paw target is Lightning Pose 2D speed, falling back to DLC.

## Holdout tasks

- **trial_holdout**: 5/52 sessions contribute held-out trials (5adab0b7, 62902992, 8c33abef, a92c4b1d, d2f5a130). train n=5707, test n=45.
- **session_holdout**: held-out session 626126d5 of CSH_ZAD_026; 454 trials from other sessions of that mouse remain in train. train n=5515, test n=237.
- **mouse_holdout**: held-out mouse ZM_2241 (one session, unseen animal). train n=5381, test n=371.

Trial holdout tests within-session decoding (neurons *are* the same cells on train and test, but the model is still not allowed to use a fixed index). Session holdout tests a new probe insertion in a mouse the model has seen. Mouse holdout tests a new animal.

## Architectures

### Shared pooling

Let `x` be `(B, N, T)` log-counts with neuron mask `m`. A shared encoder produces `h ∈ R^{B×N×T×D}`. SetPool returns `[masked_mean(h); masked_max(h); attention_pool(h)] ∈ R^{B×T×3D}`. Attention scores are `softmax_N(W h)`, with masked neurons set to −∞. Because encoder weights do not depend on the neuron index, any permutation of columns of `x` leaves the pooled sequence unchanged.

### 1. Temporal CNN (`cnn`)

Per-neuron causal Conv1d stack, channels (32, 64), kernel 5, left-padded so output time t uses bins ≤ t. Pool, Linear to 64, then two more causal convs (dilation 1 then 2). MLP head → 2.

### 2. GRU (`gru`)

Instantaneous Linear(1→64) embed per neuron, pool, Linear to 64, then a 1-layer GRU (hidden 64) on the packed variable-length sequence. Hidden state at each t → MLP head.

### 3. LSTM (`lstm`)

Identical to GRU with `nn.LSTM` in place of `nn.GRU`.

### 4. CNN–RNN (`cnn_gru`)

Same per-neuron causal CNN as (1), then pool, then the GRU of (2). CNN supplies local temporal features per cell; GRU integrates them.

### 5. Causal autoregressive transformer (`transformer`)

Instantaneous embed + pool → 64-d tokens, plus a learned positional table of length 128. 2 TransformerEncoder layers, 4 heads, FFN 128, dropout 0.1. A strictly upper-triangular attention mask makes the stack decoder-only: token t attends to 0…t (current bin included, matching BWM lag 0). Kinematics are **not** fed back, so test-time predictions come from spikes alone. That is autoregressive over neural tokens, not over past |ω|.

### 6. Ridge, fixed sliding window (`ridge`)

Each bin is reduced to four set-statistics: population mean, std, max, and fraction of neurons with a spike. A causal window of W=10 lags (11 bins × 4 = 44 features) predicts the two speeds at time t from bins t−W…t. First W bins are dropped. Features and targets are StandardScaled on the training bins. α is chosen on the validation split from [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], then Ridge is refit on train+val. This is the linear analogue of early stopping: complexity is selected on held-out MSE, not on train fit.

## Training, loss, and error

Deep models minimize **masked MSE on z-scored targets**. Train-set bin means and stds (both speeds, after the W-bin crop) standardize y so rad/s and px/s contribute on the same scale:

    L = mean_{t ≥ W, valid}  Σ_{k=1,2} (ŷ_{t,k} − ỹ_{t,k})²

AdamW (lr=0.001, weight decay=0.0001), batch 32, grad clip 1.0, max 25 epochs, early stop when val MSE does not improve for 5 epochs. Best checkpoint (lowest val MSE) is cached under `BehaviorRes/MLtestcontra/cache/<task>/<model>/` and reused.

Reported error is **not** that training loss. After inverting the y standardization we compute, on concatenated test bins with t ≥ W:

- R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ_test)²  (sklearn, original units)
- RMSE and MAE in rad/s or px/s
- Pearson r
- median per-trial R² (trials with ≥ 8 scored bins)

## Results

| task | model | target | R² | RMSE | MAE | Pearson | median trial R² | n bins | n trials |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| trial_holdout | cnn | wheel_speed | 0.246 | 0.935 | 0.757 | 0.509 | -0.188 | 566 | 24 |
| trial_holdout | cnn | paw_speed | 0.211 | 600.028 | 409.861 | 0.472 | -0.158 | 566 | 24 |
| trial_holdout | gru | wheel_speed | 0.232 | 0.944 | 0.780 | 0.496 | -0.245 | 566 | 24 |
| trial_holdout | gru | paw_speed | 0.075 | 649.846 | 424.008 | 0.284 | -0.143 | 566 | 24 |
| trial_holdout | lstm | wheel_speed | 0.024 | 1.064 | 0.912 | 0.234 | -0.215 | 566 | 24 |
| trial_holdout | lstm | paw_speed | 0.039 | 662.056 | 452.074 | 0.252 | -0.032 | 566 | 24 |
| trial_holdout | cnn_gru | wheel_speed | 0.281 | 0.914 | 0.756 | 0.560 | -0.117 | 566 | 24 |
| trial_holdout | cnn_gru | paw_speed | 0.100 | 640.889 | 433.947 | 0.318 | -0.120 | 566 | 24 |
| trial_holdout | transformer | wheel_speed | 0.201 | 0.963 | 0.813 | 0.493 | -0.243 | 566 | 24 |
| trial_holdout | transformer | paw_speed | 0.051 | 658.110 | 443.182 | 0.227 | -0.106 | 566 | 24 |
| trial_holdout | ridge | wheel_speed | 0.062 | 1.043 | 0.894 | 0.294 | -0.049 | 566 | 24 |
| trial_holdout | ridge | paw_speed | 0.114 | 635.923 | 444.099 | 0.383 | -0.136 | 566 | 24 |
| session_holdout | cnn | wheel_speed | 0.293 | 1.003 | 0.836 | 0.601 | -0.230 | 2324 | 84 |
| session_holdout | cnn | paw_speed | -0.087 | 493.046 | 391.748 | 0.436 | -1.381 | 2324 | 84 |
| session_holdout | gru | wheel_speed | 0.214 | 1.057 | 0.880 | 0.504 | -0.245 | 2324 | 84 |
| session_holdout | gru | paw_speed | -0.108 | 497.622 | 388.310 | 0.357 | -1.259 | 2324 | 84 |
| session_holdout | lstm | wheel_speed | 0.111 | 1.124 | 0.965 | 0.478 | -0.142 | 2324 | 84 |
| session_holdout | lstm | paw_speed | -0.139 | 504.510 | 397.377 | 0.357 | -0.887 | 2324 | 84 |
| session_holdout | cnn_gru | wheel_speed | 0.310 | 0.991 | 0.820 | 0.591 | -0.327 | 2324 | 84 |
| session_holdout | cnn_gru | paw_speed | -0.151 | 507.299 | 404.416 | 0.407 | -1.565 | 2324 | 84 |
| session_holdout | transformer | wheel_speed | 0.280 | 1.011 | 0.828 | 0.581 | -0.268 | 2324 | 84 |
| session_holdout | transformer | paw_speed | -0.070 | 489.094 | 384.851 | 0.408 | -1.305 | 2324 | 84 |
| session_holdout | ridge | wheel_speed | 0.093 | 1.135 | 0.963 | 0.496 | -0.129 | 2324 | 84 |
| session_holdout | ridge | paw_speed | -0.032 | 480.382 | 372.367 | 0.497 | -0.515 | 2324 | 84 |
| mouse_holdout | cnn | wheel_speed | -0.086 | 1.045 | 0.816 | 0.146 | -0.670 | 4723 | 213 |
| mouse_holdout | cnn | paw_speed | 0.023 | 634.745 | 461.301 | 0.206 | -0.344 | 4723 | 213 |
| mouse_holdout | gru | wheel_speed | -0.142 | 1.071 | 0.832 | 0.099 | -0.487 | 4723 | 213 |
| mouse_holdout | gru | paw_speed | -0.008 | 644.530 | 456.248 | 0.173 | -0.323 | 4723 | 213 |
| mouse_holdout | lstm | wheel_speed | -0.030 | 1.018 | 0.830 | 0.101 | -0.325 | 4723 | 213 |
| mouse_holdout | lstm | paw_speed | 0.010 | 638.853 | 458.948 | 0.168 | -0.189 | 4723 | 213 |
| mouse_holdout | cnn_gru | wheel_speed | -0.139 | 1.070 | 0.827 | 0.139 | -0.714 | 4723 | 213 |
| mouse_holdout | cnn_gru | paw_speed | 0.023 | 634.773 | 456.012 | 0.213 | -0.326 | 4723 | 213 |
| mouse_holdout | transformer | wheel_speed | -0.081 | 1.043 | 0.829 | 0.126 | -0.553 | 4723 | 213 |
| mouse_holdout | transformer | paw_speed | 0.024 | 634.452 | 470.590 | 0.197 | -0.293 | 4723 | 213 |
| mouse_holdout | ridge | wheel_speed | -0.054 | 1.029 | 0.825 | 0.124 | -0.249 | 4723 | 213 |
| mouse_holdout | ridge | paw_speed | -0.043 | 655.658 | 458.451 | 0.095 | -0.266 | 4723 | 213 |

### Best model per task × target (test R²)

- trial_holdout / wheel_speed: **cnn_gru** R²=0.281 (ridge=0.062)
- trial_holdout / paw_speed: **cnn** R²=0.211 (ridge=0.114)
- session_holdout / wheel_speed: **cnn_gru** R²=0.310 (ridge=0.093)
- session_holdout / paw_speed: **ridge** R²=-0.032 (ridge=-0.032)
- mouse_holdout / wheel_speed: **lstm** R²=-0.030 (ridge=-0.054)
- mouse_holdout / paw_speed: **transformer** R²=0.024 (ridge=-0.043)

## Training diagnostics

### trial_holdout

- cnn: best @ 23/25  train MSE=0.8483  val MSE=0.8395
- gru: best @ 9/14  train MSE=0.9143  val MSE=0.8846
- lstm: best @ 12/17  train MSE=0.9837  val MSE=0.9839
- cnn_gru: best @ 7/12  train MSE=0.8948  val MSE=0.8610
- transformer: best @ 11/16  train MSE=0.9178  val MSE=0.8979
- ridge: best @ 1/6  train MSE=0.9494  val MSE=0.9425

### session_holdout

- cnn: best @ 13/18  train MSE=0.8815  val MSE=0.9320
- gru: best @ 7/12  train MSE=0.9355  val MSE=0.9617
- lstm: best @ 16/21  train MSE=0.9448  val MSE=0.9849
- cnn_gru: best @ 17/22  train MSE=0.8624  val MSE=0.9211
- transformer: best @ 14/19  train MSE=0.9105  val MSE=0.9604
- ridge: best @ 4/6  train MSE=0.9514  val MSE=0.9925

### mouse_holdout

- cnn: best @ 15/20  train MSE=0.8580  val MSE=0.8826
- gru: best @ 11/16  train MSE=0.9178  val MSE=0.9151
- lstm: best @ 6/11  train MSE=0.9922  val MSE=0.9832
- cnn_gru: best @ 13/18  train MSE=0.8516  val MSE=0.8679
- transformer: best @ 25/25  train MSE=0.8849  val MSE=0.9080
- ridge: best @ 1/6  train MSE=0.9473  val MSE=0.9455

Plots: `train_<task>.png`, `r2_wheel_speed.png`, `r2_paw_speed.png`, `examples_<task>.png`.

## Interpretation

1. **CNN–GRU is the best wheel decoder that still generalizes to a new session of a known mouse** (trial R² 0.281, session R² 0.310). A temporal CNN alone is close (0.246 / 0.293) and is the only model that decodes paw at a useful level inside a seen session (trial paw R² 0.211).
2. **Paw speed does not transfer across sessions.** Pearson r stays ~0.4 on session holdout but R² goes negative: the models capture some shape and get the scale wrong. Ridge’s weaker fit is the “least bad” paw R² (−0.032) simply because it stays closer to the mean.
3. **A new mouse is out of reach.** All six models sit at R² ≈ 0 on ZM_2241. Train/val MSE still drops (the models fit other animals), so this is a generalization failure, not a failed optimization.
4. **LSTM and pooled Ridge underfit.** LSTM val MSE never leaves ~0.98–1.00 (predicting the z-scored mean). Ridge’s 44 population-lag features saturate at α = 0.1 with val MSE ~0.94. Shared per-neuron CNNs are doing work that a linear readout of mean/std/max/frac cannot.
5. **Concatenated-bin R² and per-trial R² disagree.** Median trial R² is negative even when concatenated R² is +0.3. Long trials with large |ω| swings dominate the pooled metric; short trials (the majority) are poorly fit. Example traces show models smoothing over brief speed pulses.
6. **Trial holdout is a small test set** (45 trials / 566 scored bins from 5 sessions). Session and mouse holdouts (2324 and 4723 bins) are the more stable comparisons.
7. **Training looks healthy for CNN / CNN–GRU / transformer:** val MSE tracks train and early stopping fires. Session-holdout curves show a train–val gap (val ~0.92 vs train ~0.86), consistent with a new insertion being harder than a held-out trial. Ridge’s α grid is flat — regularization is not the bottleneck; the feature set is.
