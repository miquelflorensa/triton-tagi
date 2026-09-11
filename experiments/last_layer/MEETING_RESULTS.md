# Frozen last-layer TAGI — all heads, all datasets

Protocol: frozen ResNet-18 features (512-d), 40k/10k train/val split. CIFAR: 200 epochs, 5 seeds, reported at epoch 200. ImageNet-1k: 4 epochs, 3 seeds. Every number is the seed mean. OOD = SVHN vs. test; *shift* = CIFAR-10-C / CIFAR-100-C macro over 15 corruptions × 5 severities. `epi` = native epistemic variance score.

**Which HRC row is the headline**: `hrc:full` (full K-leaf tree, 9 / 99 nodes, no prior readout offsets) with **per-node** HSM gain calibration — the learned gains are in §5. The uncalibrated padded `hrc` is kept beside it in every table. Per-node is the best NLL on both datasets but needs n ≳ 3000 fit rows; **per-level is the recommendation at a small budget** (§5.3).

---

## 1. Configuration selected per run (init + gain + sigma_v)

Selected on validation NLL over the 20-epoch screen; epoch 0 excluded from selection.

| dataset | head | mean_init | gain_w = gain_b | sigma_v | epochs | seeds |
|---|---|---|---|---|---|---|
| CIFAR-10 | `hrc` (padded, 11 nodes) | **zero** | 0.3 | 0.3 | 200 | 5 |
| CIFAR-10 | `hrc:full` (9 nodes) | **random** | 0.3 | 0.3 | 200 | 5 |
| CIFAR-10 | `remax_lognormal` | **backbone** | 0.3 | 0.3 | 200 | 5 |
| CIFAR-10 | `remax_laplace_diag` | **zero** | 0.03 | 0.3 | 200 | 5 |
| CIFAR-10 | `logit_tagiv` | random ≡ zero | 0.1 | learned | 200 | 5 |
| CIFAR-100 | `hrc` (padded, 102 nodes) | **zero** | 0.1 | 0.1 | 200 | 5 |
| CIFAR-100 | `hrc:full` (99 nodes) | **zero** | 0.1 | 0.1 | 200 | 5 |
| CIFAR-100 | `remax_lognormal` | **backbone** | 1.0 | 0.3 | 200 | 5 |
| CIFAR-100 | `remax_laplace_diag` | **zero** | 0.03 | 0.05 | 200 | 5 |
| CIFAR-100 | `logit_tagiv` | random ≡ zero | 0.3 | learned | 200 | 5 |
| ImageNet-1k | `hrc` (padded, 1001 nodes) | **zero** | 0.3 | 0.3 | 4 | 3 |
| ImageNet-1k | `remax_lognormal` | **backbone** | 0.1 | 0.1 | 4 | 3 |
| ImageNet-1k | `remax_laplace_diag` | **backbone** | 0.1 | 0.1 | 4 | 3 |
| ImageNet-1k | `logit_tagiv` | **backbone** | 1.0 | learned | 4 | 3 |

`scale = sqrt(1/512) = 0.04419`, `Sw = Sb = (gain·scale)²`. `logit_tagiv` learns its observation noise and its `random` / `zero` arms are bit-identical (its prior zeroes the latent means by construction). `zero` is the only arm that is flat in gain — it removes a tuning axis; `random` spans a 7× NLL range over the gain grid and `backbone` wants a *tight* prior, not a loose one.

---

## 2. CIFAR-10 — accuracy, UQ, OOD

| head | init | gain | sigma_v | top-1 | top-5 | NLL | ECE | Brier | AURC | mean conf | AUROC ent | FPR95 ent | AUROC maxp | AUROC epi† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hrc:full` + **node** calib. | random | 0.3 | 0.3 | 0.9474 | 0.9974 | **0.1783** | 0.0060 | 0.0791 | 0.0057 | 0.9493 | 0.9222 | 0.2109 | 0.9197 | 0.9035 |
| `hrc:full` + level calib. | random | 0.3 | 0.3 | 0.9473 | 0.9968 | 0.1799 | 0.0058 | 0.0791 | 0.0057 | 0.9495 | 0.9182 | 0.2229 | 0.9165 | 0.9053 |
| `hrc:full` + global calib. | random | 0.3 | 0.3 | 0.9472 | 0.9956 | 0.1873 | 0.0059 | 0.0793 | 0.0056 | 0.9502 | 0.9191 | 0.2179 | 0.9159 | 0.9117 |
| `hrc:full` uncalibrated | random | 0.3 | 0.3 | 0.9473 | 0.9956 | 0.1875 | 0.0053 | 0.0794 | 0.0056 | 0.9512 | 0.9189 | 0.2181 | 0.9158 | 0.1383 |
| `hrc` (padded) | zero | 0.3 | 0.3 | 0.9481 | 0.9964 | 0.1806 | **0.0048** | 0.0782 | 0.0056 | 0.9518 | 0.9238 | 0.2072 | 0.9212 | 0.2374 |
| `hrc` (padded) | random | 0.3 | 0.3 | 0.9483 | 0.9964 | 0.1814 | 0.0053 | 0.0784 | 0.0056 | 0.9518 | **0.9248** | **0.2062** | **0.9221** | 0.2374 |
| `logit_tagiv` | random ≡ zero | 0.1 | — | **0.9499** | 0.9980 | 0.1939 | 0.0278 | 0.0799 | 0.0057 | 0.9776 | 0.9116 | 0.2412 | 0.9097 | 0.0914 |
| `remax_laplace_diag` | zero | 0.03 | 0.3 | 0.9495 | 0.9935 | 0.2595 | 0.0293 | 0.0833 | 0.0068 | 0.9766 | 0.8652 | 0.6737 | 0.8636 | 0.8204 |
| `remax_laplace_diag` | random | 0.03 | 0.3 | 0.5297 | 0.7447 | 1.6877 | 0.0714 | 0.4916 | 0.1782 | 0.5595 | 0.4629 | 0.9329 | 0.4695 | 0.5893 |
| `remax_lognormal` | backbone | 0.3 | 0.3 | 0.9490 | 0.9843 | 0.3943 | 0.0276 | 0.0825 | 0.0174 | 0.9766 | 0.6824 | 0.9180 | 0.6060 | 0.5484 |
| `remax_lognormal` | random | 0.3 | 0.3 | 0.9216 | 0.9546 | 0.4548 | 0.0286 | 0.1067 | 0.0228 | 0.9466 | 0.6205 | 0.9944 | 0.6429 | 0.8544 |
| **`pytorch_softmax`** (ref.) | — | — | — | 0.9500 | 0.9980 | 0.1941 | 0.0278 | 0.0800 | 0.0057 | 0.9777 | 0.9118 | 0.2401 | 0.9100 | — |
| **tempered softmax** (ref., T = 1.3487) | — | — | — | 0.9500 | 0.9980 | 0.1722 | 0.0102 | 0.0764 | 0.0059 | 0.9600 | 0.9120 | 0.2524 | 0.9098 | — |

† The native-epistemic column is **not comparable across heads or between calibrated and uncalibrated rows** — see §6.4.

## 3. CIFAR-100 — accuracy, UQ, OOD

| head | init | gain | sigma_v | top-1 | top-5 | NLL | ECE | Brier | AURC | mean conf | AUROC ent | FPR95 ent | AUROC maxp | AUROC epi† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hrc:full` + **node** calib. | zero | 0.1 | 0.1 | 0.7284 | 0.8998 | **1.1576** | 0.0904 | 0.3917 | 0.0885 | 0.6381 | 0.8559 | 0.4566 | 0.8159 | 0.8009 |
| `hrc:full` + level calib. | zero | 0.1 | 0.1 | 0.7305 | 0.8986 | 1.1627 | 0.0931 | 0.3910 | 0.0877 | 0.6374 | **0.8618** | **0.4426** | 0.8245 | 0.7269 |
| `hrc:full` + global calib. | zero | 0.1 | 0.1 | 0.7301 | 0.8750 | 1.2813 | 0.0929 | 0.3922 | 0.0829 | 0.6371 | 0.8583 | 0.4468 | 0.8332 | 0.6568 |
| `hrc:full` uncalibrated | zero | 0.1 | 0.1 | 0.7297 | 0.8747 | 1.2810 | 0.0881 | 0.3914 | 0.0830 | 0.6416 | 0.8578 | 0.4486 | 0.8326 | 0.1223 |
| `hrc:full` uncalibrated | random | 0.1 | 0.1 | 0.7268 | 0.8699 | 1.3035 | 0.0880 | 0.3946 | 0.0837 | 0.6389 | 0.8580 | 0.4480 | 0.8346 | 0.1223 |
| `hrc` (padded) | zero | 0.1 | 0.1 | 0.7302 | 0.8761 | 1.2816 | 0.0877 | 0.3906 | 0.0825 | 0.6425 | 0.8437 | 0.4665 | 0.8280 | 0.1201 |
| `hrc` (padded) | random | 0.1 | 0.1 | 0.7270 | 0.8714 | 1.3051 | 0.0874 | 0.3942 | 0.0837 | 0.6397 | 0.8397 | 0.4735 | 0.8256 | 0.1201 |
| `logit_tagiv` | random ≡ zero | 0.3 | — | **0.7673** | **0.9363** | 0.9566 | **0.0462** | **0.3330** | **0.0668** | 0.7773 | 0.8589 | 0.4319 | **0.8414** | 0.1166 |
| `remax_laplace_diag` | zero | 0.03 | 0.05 | 0.7588 | 0.8777 | 1.7479 | 0.1306 | 0.3873 | 0.0791 | 0.7261 | 0.7897 | 0.6094 | 0.7858 | 0.6184 |
| `remax_laplace_diag` | random | 0.03 | 0.05 | 0.3797 | 0.4136 | 3.8594 | 0.0570 | 0.6790 | 0.2968 | 0.4177 | 0.6848 | 0.6027 | 0.6908 | 0.3719 |
| `remax_lognormal` | backbone | 1.0 | 0.3 | 0.7484 | 0.8115 | 2.1180 | 0.1405 | 0.4006 | 0.1107 | 0.8865 | 0.6743 | 0.8120 | 0.6531 | 0.6531 |
| `remax_lognormal` | random | 1.0 | 0.3 | 0.7478 | 0.8107 | 2.1103 | 0.1413 | 0.4017 | 0.1088 | 0.8870 | 0.6765 | 0.8247 | 0.6577 | 0.6651 |
| **`pytorch_softmax`** (ref.) | — | — | — | 0.7667 | 0.9362 | 0.9581 | 0.0482 | 0.3329 | 0.0662 | 0.7840 | 0.8549 | 0.4394 | 0.8378 | — |
| **tempered softmax** (ref., T = 0.9794) | — | — | — | 0.7667 | 0.9362 | 0.9578 | 0.0495 | 0.3330 | 0.0660 | 0.7902 | 0.8541 | 0.4416 | 0.8369 | — |

Temperature scaling buys CIFAR-100 nothing (T ≈ 1): the softmax baseline is already at its optimal temperature there.

## 4. ImageNet-1k — accuracy and UQ (no OOD set cached)

4 epochs, 3 seeds, 1000 classes. The OOD columns are a CIFAR-only result — nothing here shows whether they hold at 1000 classes.

| head | init | gain | sigma_v | top-1 | top-5 | NLL | ECE | Brier | AURC | mean conf |
|---|---|---|---|---|---|---|---|---|---|---|
| `logit_tagiv` | backbone | 1.0 | — | **0.6976** | **0.8908** | **1.2469** | **0.0263** | **0.4111** | 0.1014 | 0.7202 |
| `remax_lognormal` | backbone | 0.1 | 0.1 | 0.6892 | 0.8847 | 4.2153 | 0.6722 | 0.9724 | 0.1515 | 0.0170 |
| `remax_laplace_diag` | backbone | 0.1 | 0.1 | 0.6869 | 0.8837 | 4.2187 | 0.6698 | 0.9723 | 0.1530 | 0.0171 |
| `hrc` (padded) | zero | 0.3 | 0.3 | 0.4331 | 0.6860 | 2.7013 | 0.0891 | 0.7259 | 0.3318 | 0.3443 |
| `hrc` (padded) | random | 0.3 | 0.3 | 0.3899 | 0.6349 | 3.0524 | 0.0546 | 0.7631 | 0.3782 | 0.3360 |
| `remax_lognormal` | random | 0.1 | 0.1 | 0.1958 | 0.4247 | 5.5642 | 0.1895 | 0.9932 | 0.6768 | 0.0063 |
| `remax_laplace_diag` | random | 0.1 | 0.1 | 0.0472 | 0.0517 | 6.7471 | 0.0166 | 0.9704 | 0.8211 | 0.0493 |
| **`pretrained_fc` softmax** (ref.) | — | — | — | 0.6976 | 0.8908 | 1.2469 | 0.0263 | 0.4111 | 0.1014 | 0.7202 |
| **tempered softmax** (ref., T = 1.0873) | — | — | — | 0.6976 | 0.8908 | 1.2400 | 0.0184 | 0.4111 | 0.1018 | 0.6936 |

Three things at 1000 classes: (i) `random` means collapse the remax heads (0.196 / 0.047 top-1) — **+49 / +64 points from `zero` or `backbone` alone**; (ii) the remax heads rank well and report near-uniform probability (mean confidence 0.017), so accuracy and NLL rank them almost independently; (iii) `hrc` degrades badly with class count — this is a scaling result about the flat tree, and §6.2 shows a hidden layer fixes most of it. The ImageNet tempered-softmax temperature is fitted on the same 50k rows it is scored on (no held-out split exists there).

---

## 5. Which HRC calibration, and the gains it learns

Hierarchical-probit gain calibration (Goulet / Nguyen / Florensa-Montilla): a Gaussian belief over each group's positive-branch **log gain**, fitted on the 10 000-row validation split with the network frozen, integrated over at prediction. `hrc:full` only — `gain_groups` refuses the padded tree. The `hrc:full+offsets` arm — the same tree trained with the uncompensated prior readout bias `run_study` took by default — is a measured ablation and is excluded from the tables above; it is uniformly ~0.01 (CIFAR-10) to ~0.03 (CIFAR-100) nats worse.

### 5.1 What each sharing level buys (5 seeds)

| dataset | arm | init | sharing | groups | NLL | ΔNLL vs uncal. | ECE | top-1 | AUROC ent |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | `hrc:full` | random | uncalibrated | — | 0.1875 | — | 0.0053 | 0.9473 | 0.9189 |
| CIFAR-10 | `hrc:full` | random | global | 1 | 0.1873 | −0.0002 | 0.0059 | 0.9472 | 0.9191 |
| CIFAR-10 | `hrc:full` | random | level | 4 | 0.1799 | −0.0076 | 0.0058 | 0.9473 | 0.9182 |
| CIFAR-10 | `hrc:full` | random | **node** | 9 | **0.1783** | **−0.0092** | 0.0060 | 0.9474 | 0.9222 |
| CIFAR-100 | `hrc:full` | zero | uncalibrated | — | 1.2810 | — | 0.0881 | 0.7297 | 0.8578 |
| CIFAR-100 | `hrc:full` | zero | global | 1 | 1.2813 | +0.0003 | 0.0929 | 0.7301 | 0.8583 |
| CIFAR-100 | `hrc:full` | zero | level | 7 | 1.1627 | −0.1183 | 0.0931 | 0.7305 | 0.8618 |
| CIFAR-100 | `hrc:full` | zero | **node** | 99 | **1.1576** | **−0.1234** | 0.0904 | 0.7284 | 0.8559 |
| CIFAR-10 | `pytorch_softmax` (ref.) | — | — | — | 0.1941 | — | 0.0278 | 0.9500 | 0.9118 |
| CIFAR-10 | tempered softmax (ref.) | — | — | — | **0.1722** | — | 0.0102 | 0.9500 | 0.9120 |
| CIFAR-100 | `pytorch_softmax` (ref.) | — | — | — | 0.9581 | — | 0.0482 | 0.7667 | 0.8549 |
| CIFAR-100 | tempered softmax (ref.) | — | — | — | **0.9578** | — | 0.0495 | 0.7667 | 0.8541 |

**A global gain does nothing** (±0.0003 on both datasets) — the α = 3 convention already sits at the global optimum. The gain belief only pays once it can vary *across* the tree. It buys likelihood, not bin-wise calibration: ECE is flat or slightly worse at every level. And softmax temperature scaling still beats all of it on NLL — say so before a supervisor finds it.

### 5.2 The gains that are actually learned (posterior mean of the log gain, seed-averaged)

| dataset | sharing | groups | learned gain `exp(mu)` | posterior sd of `mu` |
|---|---|---|---|---|
| CIFAR-10 | global | 1 | 0.892 | 0.012 |
| CIFAR-10 | level | 4 | 0.776, 1.030, 1.002, 0.986 (root → leaves) | 0.018 – 0.043 |
| CIFAR-10 | node | 9 | min 0.776, median 1.067, max 1.450 | 0.018 – 0.148 |
| CIFAR-100 | global | 1 | 0.293 | 0.007 |
| CIFAR-100 | level | 7 | 0.188, 0.240, 0.298, 0.399, 0.500, 0.436, 0.629 (root → leaves) | 0.016 – 0.038 |
| CIFAR-100 | node | 99 | min 0.188, median 0.560, max 2.222 | 0.016 – 0.746 |

Read: on CIFAR-10 the fitted gain is ≈ 1 everywhere — the uncalibrated head is already right, which is why calibration buys 0.009 nats. On CIFAR-100 it is **0.29 globally and rises monotonically with depth** (0.19 at the root to 0.63 at the leaves): the trained head is systematically over-sharp near the root, and the correction is depth-structured, which is exactly the structure a single global gain cannot express. Per-node posterior sd reaches 0.75 on CIFAR-100 — 99 groups against 10 000 rows is the data-starved edge of the method.

### 5.3 How much fit data each level needs (test NLL vs. fit rows)

| dataset | method | groups | n=100 | n=300 | n=1000 | n=3000 | n=10000 |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | uncalibrated | — | 0.1893 | 0.1893 | 0.1893 | 0.1893 | 0.1893 |
| CIFAR-10 | `hsm_global` | 1 | 0.1956 | 0.1899 | 0.1901 | 0.1896 | 0.1891 |
| CIFAR-10 | `hsm_level` | 4 | 0.2043 | 0.1869 | 0.1845 | 0.1825 | 0.1818 |
| CIFAR-10 | `hsm_node` | 9 | 0.2123 | 0.1910 | 0.1852 | 0.1814 | **0.1802** |
| CIFAR-100 | uncalibrated | — | 1.3150 | 1.3150 | 1.3150 | 1.3150 | 1.3150 |
| CIFAR-100 | `hsm_global` | 1 | 1.3170 | 1.3152 | 1.3149 | 1.3153 | 1.3149 |
| CIFAR-100 | `hsm_level` | 7 | 1.2301 | 1.2092 | 1.2035 | 1.2012 | 1.2003 |
| CIFAR-100 | `hsm_node` | 99 | 1.4105 | 1.2882 | 1.2268 | 1.2044 | **1.1943** |
| CIFAR-10 | tempered softmax (ref.) | — | 0.1922 | **0.1738** | **0.1739** | **0.1736** | **0.1734** |
| CIFAR-100 | tempered softmax (ref.) | — | **0.9559** | **0.9555** | **0.9541** | **0.9541** | **0.9540** |

The crossover is real: per-node is **worse than uncalibrated** below n ≈ 1000 (1.4105 vs 1.3150 at n=100 on CIFAR-100 — one visit per node at 99 groups) and best above n ≈ 3000. **`hsm_level` is the recommendation at a fixed budget**: it beats uncalibrated from n=300 up on both datasets, and beats per-node everywhere below n ≈ 3000, with 4 and 7 groups.

---

## 6. Extra linear layer at the end of the head

`Linear(512→H) + ReLU` in front of the head's output layer. CIFAR-100: 20 epochs, seed 0, selection on validation NLL over a 16-cell gain × sigma_v grid per arm. ImageNet: 1 epoch, seed 0, screen only.

### 6.1 CIFAR-100 (test)

| head | init | depth | gain | sigma_v | top-1 | Δ top-1 | NLL | Δ NLL | ECE | params |
|---|---|---|---|---|---|---|---|---|---|---|
| `hrc` | zero | flat | 0.1 | 0.1 | 0.7302 | — | 1.2855 | — | 0.0908 | 104,652 |
| `hrc` | zero | **+512** | 0.1 | 0.05 | **0.7539** | **+0.0237** | **1.0707** | **−0.2148** | **0.0512** | 629,964 |
| `hrc` | random | flat | 0.3 | 0.1 | 0.7248 | — | 1.3115 | — | 0.0868 | 104,652 |
| `hrc` | random | **+512** | 0.1 | 0.1 | **0.7530** | **+0.0282** | **1.1047** | **−0.2069** | 0.0527 | 629,964 |
| `remax_laplace_diag` | zero | flat | 0.03 | 0.05 | 0.7612 | — | 1.5967 | — | 0.1230 | 102,600 |
| `remax_laplace_diag` | zero | +512 | 0.3 | 0.03 | 0.7640 | +0.0028 | 1.4440 | −0.1526 | 0.1018 | 627,912 |
| `remax_laplace_diag` | random | flat | 1.0 | 0.1 | 0.7542 | — | 1.5783 | — | 0.1133 | 102,600 |
| `remax_laplace_diag` | random | +512 | 1.0 | 0.3 | 0.7603 | +0.0061 | 1.7379 | +0.1596 | 0.1245 | 627,912 |
| `remax_lognormal` | zero | flat | 1.0 | 0.05 | 0.7352 | — | 1.8380 | — | 0.1335 | 102,600 |
| `remax_lognormal` | zero | +512 | 1.0 | 0.3 | 0.7403 | +0.0051 | 1.8154 | −0.0226 | 0.1274 | 627,912 |
| `remax_lognormal` | random | flat | 0.03 | 0.1 | 0.7570 | — | 1.6577 | — | 0.4026 | 102,600 |
| `remax_lognormal` | random | +512 | 1.0 | 0.3 | 0.7488 | −0.0082 | 1.8133 | +0.1555 | 0.1337 | 627,912 |
| **deterministic MAP** (ref.) | — | flat | — | — | 0.7649 | — | 1.0171 | — | — | — |
| **deterministic MAP** (ref.) | — | +512 | — | — | 0.7646 | −0.0003 | 1.1795 | +0.1624 | — | — |
| **`pytorch_softmax`** (ref.) | — | flat | — | — | 0.7667 | — | 0.9581 | — | 0.0482 | — |
| **tempered softmax** (ref.) | — | flat | — | — | 0.7667 | — | 0.9578 | — | 0.0495 | — |

**Depth fixes `hrc`, not Remax.** `hrc` gains +2.4 to +2.8 points and 0.21 nats and halves its ECE, closing to within 1.1 points of the deterministic ceiling; the Remax family gains nothing beyond noise. The deterministic MAP reference is **0.7649 flat vs 0.7646 with the hidden layer** — depth buys a deterministic model nothing on these frozen features, so this is not a capacity argument: `hrc` asks a linear map to score tree nodes over arbitrary *class subsets*, which the backbone never made linearly separable, while Remax's per-class logits are exactly what the deterministic softmax already computes.

### 6.2 ImageNet-1k (validation, 1 epoch, seed 0 — screen only)

| head | init | depth | gain | sigma_v | top-1 | Δ top-1 | top-5 | NLL | ECE |
|---|---|---|---|---|---|---|---|---|---|
| `hrc` | zero | flat | 0.3 | 0.3 | 0.4260 | — | 0.6804 | 2.7374 | 0.0928 |
| `hrc` | zero | +512 | 0.3 | 0.1 | 0.5660 | +0.1400 | 0.7867 | 2.0470 | 0.0587 |
| `hrc` | zero | +1024 | 0.3 | 0.1 | 0.5946 | +0.1686 | 0.8076 | 1.8952 | 0.0471 |
| `hrc` | zero | **+2048** | 0.3 | 0.1 | **0.6168** | **+0.1908** | **0.8221** | **1.7850** | **0.0398** |
| `hrc` | random | flat | 1.0 | 0.3 | 0.3721 | — | 0.6141 | 3.2046 | 0.0506 |
| `hrc` | random | +2048 | 0.3 | 0.3 | 0.5824 | +0.2103 | 0.7958 | 1.9703 | 0.0691 |
| `remax_lognormal` | zero | flat | 0.3 | 0.1 | 0.5428 | — | 0.8062 | 5.1915 | 0.5362 |
| `remax_lognormal` | zero | +512 | 0.1 | 0.1 | 0.5083 | −0.0345 | 0.7719 | 5.6453 | 0.5043 |
| `remax_lognormal` | random | flat | 1.0 | 0.1 | 0.5165 | — | 0.7906 | 5.0685 | 0.5089 |
| `remax_lognormal` | random | +512 | 0.3 | 0.1 | 0.4283 | −0.0882 | 0.6872 | 5.0504 | 0.4201 |
| **`pretrained_fc` softmax** (ref.) | — | flat | — | — | 0.6976 | — | 0.8908 | 1.2469 | 0.0263 |
| **tempered softmax** (ref.) | — | flat | — | — | 0.6976 | — | 0.8908 | 1.2400 | 0.0184 |

The same split: the hidden layer recovers **+19 points of `hrc` top-1 at 1000 classes** and is still improving at width 2048, while `remax_lognormal` gets *worse* with depth. This is 1 epoch, one seed, on validation — a confirm run is not done.

---

## 7. OOD detection — all models side by side

SVHN vs. test (AUROC ↑, FPR95 ↓) under three scores, plus the CIFAR-C corruption-shift detection macro. Same frozen ResNet-18 features for every row.

### 7.1 CIFAR-10

| model | init | entropy AUROC | entropy FPR95 | max-prob AUROC | epi AUROC† | shift AUROC (ent) | shift top-1 | shift NLL |
|---|---|---|---|---|---|---|---|---|
| `hrc` (padded) | random | **0.9248** | **0.2062** | **0.9221** | 0.2374 | 0.7133 | 0.7189 | 1.1284 |
| `hrc` (padded) | zero | 0.9238 | 0.2072 | 0.9212 | 0.2374 | 0.7134 | 0.7192 | 1.1236 |
| `hrc:full` + node calib. | random | 0.9222 | 0.2109 | 0.9197 | 0.9035 | 0.7025 | 0.7198 | 1.1449 |
| `hrc:full` uncalibrated | random | 0.9189 | 0.2181 | 0.9158 | 0.1383 | **0.7134** | 0.7196 | 1.1261 |
| `logit_tagiv` | random ≡ zero | 0.9116 | 0.2412 | 0.9097 | 0.0914 | 0.7077 | 0.7212 | 1.3556 |
| `remax_laplace_diag` | zero | 0.8652 | 0.6737 | 0.8636 | 0.8204 | 0.6810 | **0.7221** | 1.6188 |
| `remax_lognormal` | backbone | 0.6824 | 0.9180 | 0.6060 | 0.5484 | 0.6266 | 0.7221 | 2.8794 |
| `remax_lognormal` | random | 0.6205 | 0.9944 | 0.6429 | 0.8544 | 0.6036 | 0.6486 | 2.1932 |
| `remax_laplace_diag` | random | 0.4629 | 0.9329 | 0.4695 | 0.5893 | 0.5238 | 0.4113 | 3.9957 |
| **`pytorch_softmax`** (ref.) | — | 0.9118 | 0.2401 | 0.9100 | — | 0.7076 | 0.7213 | 1.3578 |
| **tempered softmax** (ref.) | — | 0.9120 | 0.2524 | 0.9098 | — | 0.7080 | 0.7213 | 1.0919 |

### 7.2 CIFAR-100

| model | init | entropy AUROC | entropy FPR95 | max-prob AUROC | epi AUROC† | shift AUROC (ent) | shift top-1 | shift NLL |
|---|---|---|---|---|---|---|---|---|
| `hrc:full` + level calib. | zero | **0.8618** | **0.4426** | 0.8245 | 0.7269 | 0.6811 | 0.4392 | 2.8358 |
| `logit_tagiv` | random ≡ zero | 0.8589 | 0.4319 | **0.8414** | 0.1166 | **0.6936** | **0.4804** | 2.4464 |
| `hrc:full` + global calib. | zero | 0.8583 | 0.4468 | 0.8332 | 0.6568 | 0.6844 | 0.4400 | 2.9930 |
| `hrc:full` uncalibrated | zero | 0.8578 | 0.4486 | 0.8326 | 0.1223 | 0.6844 | 0.4396 | 3.0065 |
| `hrc:full` + node calib. | zero | 0.8559 | 0.4566 | 0.8159 | 0.8009 | 0.6809 | 0.4380 | 2.8816 |
| `hrc` (padded) | zero | 0.8437 | 0.4665 | 0.8280 | 0.1201 | 0.6791 | 0.4418 | 3.0598 |
| `hrc` (padded) | random | 0.8397 | 0.4735 | 0.8256 | 0.1201 | 0.6786 | 0.4383 | 3.1057 |
| `remax_laplace_diag` | zero | 0.7897 | 0.6094 | 0.7858 | 0.6184 | 0.6785 | 0.4677 | 3.4221 |
| `remax_lognormal` | random | 0.6765 | 0.8247 | 0.6577 | 0.6651 | 0.6237 | 0.4515 | 4.8526 |
| `remax_lognormal` | backbone | 0.6743 | 0.8120 | 0.6531 | 0.6531 | 0.6234 | 0.4516 | 4.8812 |
| `remax_laplace_diag` | random | 0.6848 | 0.6027 | 0.6908 | 0.3719 | 0.5891 | 0.1978 | 4.8473 |
| **`pytorch_softmax`** (ref.) | — | 0.8549 | 0.4394 | 0.8378 | — | 0.6939 | 0.4811 | 2.4517 |
| **tempered softmax** (ref.) | — | 0.8541 | 0.4416 | 0.8369 | — | 0.6937 | 0.4811 | 2.4638 |

### 7.3 The backbone moves OOD more than the head does

Deterministic softmax on two backbones, identical data / transform / OOD source:

| backbone | dim | top-1 | NLL | ECE | entropy AUROC | entropy FPR95 | max-prob AUROC |
|---|---|---|---|---|---|---|---|
| ResNet-18 (study) | 512 | 0.9500 | 0.1941 | 0.0278 | 0.9118 | 0.2401 | 0.9100 |
| RepVGG-A2 (pretrained) | 1408 | 0.9486 | 0.2259 | 0.0346 | **0.9557** | **0.1460** | **0.9500** |

Swapping the backbone is worth **+0.044 AUROC / −0.094 FPR95** — larger than the entire spread between the best and worst TAGI head that trains at all. And the confound runs backwards: RepVGG is *worse* in-distribution on accuracy, NLL and ECE, so "the better OOD model is just the better model" is ruled out. Caveat: the pretrained RepVGG saw all 50k CIFAR-10 train images, so no ID column from the study's validation split is comparable for it; SVHN is unseen by both, so the OOD comparison stands. Every OOD number above is a joint statement about head **and** backbone.

### 7.4 The native-epistemic column does not measure uncertainty here

A frozen *deterministic* backbone hands the head zero input variance, so the TAGI output variance collapses to `Sz = ma² @ Sw + Sb` — a monotone function of the feature energy `‖f‖²`. Measured on CIFAR-10 test: spearman(epistemic score, `‖f‖²`) = **+0.9988** for ProbiTree and +0.7136 for `hrc`; AUROC of `‖f‖²` alone is 0.1149, the same number for both heads because it is the same features. SVHN's features are *smaller* than CIFAR-10's (mean `‖f‖²` 24.04 vs 33.04), so a score that rises with feature norm points the wrong way. Over 75 CIFAR-10-C shards the epistemic AUROC is below 0.5 in **75/75**, and the *more* a corruption destroys accuracy the *more* confidently the score reports reduced uncertainty (spearman +0.94); entropy behaves correctly on the same runs. Two further reasons not to read this column down a table: it is **node space for `hrc` and class-probability space for the calibrated rows**, so the 0.14 → 0.90 jump on calibration is a change of score, not an effect; and it is not comparable across heads. Recovering a real epistemic column needs input variance to reach the head — an unfrozen or stochastic backbone — which is a different study.

---

## 8. Gaps to state out loud

1. No ImageNet OOD set is cached (decided, not overlooked) — every OOD number is CIFAR-only.
2. The gain belief is fitted on the same 10 000-row validation split that selected the cell.
3. `hrc:full` has no `zero`-init confirm on CIFAR-10 and no `backbone`-init confirm anywhere, so "does calibration absorb the init delta" is answerable on CIFAR-100 only (it does not: node calibration moves both init arms by the same 0.12 nats).
4. The hidden-layer arm is 20 epochs / 1 seed on CIFAR-100 and 1 epoch / 1 seed on ImageNet; neither is a confirm run, and neither has OOD.
5. `remax_laplace_diag` trains at batch 134 on ImageNet against 256 for every other head — its Laplace Jacobian is O(K²) and OOMs otherwise.
6. Selection on validation NLL prefers underconfident cells on CIFAR-100 (e.g. `remax_lognormal` backbone gain 0.03: NLL 1.582, ECE 0.440, mean confidence 0.3235 at 76% accuracy). For a calibration study, NLL with Brier and ECE only as tiebreakers rewards hedging.
