# 精简实验设计分析报告

## 保留的四个主实验

| rank | experiment | contrast | effect_summary | comparability | retained_reason |
| --- | --- | --- | --- | --- | --- |
| 1 | exact_polynomial_spectrum_alignment | Polynomial spectrum signal-strength threshold | range by alpha: alpha=0.2 0.6196; alpha=0.5 0.0805; alpha=0.8 0.0576 | Strong within each alpha: d=20000, n=100, mu direction fixed; only ||mu||_2 changes. | Most theory-aligned signal-threshold contrast. |
| 2 | exact_mu_eigendirection_ablation | Mu eigendirection ablation | A_svm_wstar means: largest 0.9966, smallest 0.8336, random 0.6428 | Very strong paired design: same d,n,Sigma,||mu||_2 and design seed; only mu direction changes. | Cleanest anisotropic geometry ablation. |
| 3 | exact_spectrum_scaling_alignment | Spectrum scaling: fixed-rate d-sweep and fixed-d n-sweep | fixed-d sample-size ranges from 0.0013 to 0.5771; fixed-rate d=20000 means span 0.7954 to 0.9974 | Fixed-rate panel compares spectra along n/d=0.05; fixed-d panel compares n at d=20000. Non-identity Sigma uses ||mu||_2 fixed at 2sqrt(log 20000). | Replaces the weak fixed-n varying-d design and adds polynomial spectra to fixed-rate scaling. |
| 4 | loss_comparison_anisotropic | Loss function x covariance spectrum | square-logistic final delta ranges from -0.0385 to -0.0136; polynomial alpha=0.8 delta -0.0156 | Very strong within each Sigma: same seed, data and checkpoints; only loss update changes. | Only retained algorithmic baseline for optimization-bias effects. |

## 变量可比较性审计

| experiment_or_contrast | intended_factor | controlled_variables | pairing | absorbed_old_experiments | caveat |
| --- | --- | --- | --- | --- | --- |
| loss_comparison_anisotropic | loss function within each Sigma; Sigma family across panels | within each sigma_label: d=20000; n=1000; mu_direction=largest_eigenvector; mu_norm_l2; seed; design matrix; checkpoints | paired logistic vs square by seed and design | loss_comparison_isotropic; logistic GD sample-size path sanity checks | cross-Sigma comparisons also change mu_norm_sigma and theorem diagnostics |
| exact_spectrum_scaling_alignment | fixed-rate d-scaling and fixed-d sample-size scaling across spectra | fixed-rate: n/d=0.05; fixed-d: d=20000; non-identity Sigma uses ||mu||_2=2sqrt(log 20000) | matched by d grid in fixed-rate panel and n grid in fixed-d panel | exact_isotropic_alignment_varying_d; polynomial Panel A fixed-n varying-d | cross-spectrum comparisons intentionally change Sigma spectrum and mu Sigma-norm diagnostics |
| exact_polynomial_spectrum_alignment | mu_norm_l2 within each alpha | d=20000; n=100; mu_direction=largest_eigenvector; exact SVM solver | not paired across mu_norm_l2 | exact_theorem_alignment_trends signal-strength component | cross-alpha comparison intentionally changes Sigma spectrum |
| exact_mu_eigendirection_ablation | mu direction relative to Sigma eigenvectors | d=20000; n=100; alpha=0.8; Sigma; mu_norm_l2; exact SVM solver; design seed | paired by design seed across directions | standalone SVM-LS equivalence diagnostic | random direction adds a controlled extra random mu draw |

## 关键数值证据

### Loss function paired deltas

| sigma_label | mean | std | min | max | count |
| --- | --- | --- | --- | --- | --- |
| isotropic | -0.0136 | 0.0007 | -0.0157 | -0.0117 | 100 |
| low_rank | -0.0184 | 0.0010 | -0.0211 | -0.0162 | 100 |
| polynomial alpha=0.2 | -0.0280 | 0.0013 | -0.0310 | -0.0243 | 100 |
| polynomial alpha=0.5 | -0.0385 | 0.0018 | -0.0433 | -0.0341 | 100 |
| polynomial alpha=0.8 | -0.0156 | 0.0018 | -0.0193 | -0.0111 | 100 |

### Fixed-d sample-size ranges

| sigma_label | min | max | effect_range |
| --- | --- | --- | --- |
| isotropic | 0.2997 | 0.8768 | 0.5771 |
| polynomial alpha=0.2 | 0.8674 | 0.9810 | 0.1135 |
| polynomial alpha=0.5 | 0.9950 | 0.9971 | 0.0021 |
| polynomial alpha=0.8 | 0.9964 | 0.9977 | 0.0013 |

### Fixed-rate panel at d=20000

| sigma_label | d | n | mean | std | count |
| --- | --- | --- | --- | --- | --- |
| isotropic | 20000 | 1000 | 0.7954 | 0.0021 | 20 |
| polynomial alpha=0.2 | 20000 | 1000 | 0.9766 | 0.0007 | 20 |
| polynomial alpha=0.5 | 20000 | 1000 | 0.9970 | 0.0004 | 20 |
| polynomial alpha=0.8 | 20000 | 1000 | 0.9974 | 0.0009 | 20 |

### Polynomial signal-strength ranges

| alpha | min | max | effect_range |
| --- | --- | --- | --- |
| 0.2 | 0.3557 | 0.9753 | 0.6196 |
| 0.5 | 0.9178 | 0.9983 | 0.0805 |
| 0.8 | 0.9408 | 0.9983 | 0.0576 |

### Mu eigendirection means

| mu_direction | mean | std | count |
| --- | --- | --- | --- |
| largest_eigenvector | 0.9966 | 0.0021 | 50 |
| random | 0.6428 | 0.0156 | 50 |
| smallest_eigenvector | 0.8336 | 0.0344 | 50 |

## 理论诊断量对比

Diagnostics contrast 不是新的数据生成实验，而是对四个保留实验的 CSV 做二次汇总：按每个实验的核心对照维度分组，同时列出方向对齐均值与 `M_trace`、`M_mu`、`T` 等理论诊断量，用来判断观察到的方向差异是否伴随理论控制量变化。

### Diagnostic ranges by retained contrast

| context | groups | alignment_range | M_trace_range | M_mu_range | T_range |
| --- | --- | --- | --- | --- | --- |
| E1 final checkpoint by Sigma and loss | 10 | 0.7943..1.0000 | 0.0010..0.1414 | 4.4505..6.2940 | 15.1949..38.6364 |
| E2 spectrum scaling by panel, Sigma and d/n | 56 | 0.2997..0.9977 | 0.0002..2.8284 | 6.2940..6.2940 | 3.5615..38.6378 |
| E3 polynomial signal threshold by alpha and ||mu||_2 | 21 | 0.3557..0.9983 | 0.0123..1.3703 | 1.0000..12.0000 | 0.1200..142.9844 |
| E4 mu eigendirection by direction | 3 | 0.6428..0.9966 | 0.0235..0.0318 | 6.2940..330.6253 | 38.6169..1513.0265 |

### Alignment correlations with diagnostics

| context | diagnostic | groups | corr_alignment | corr_alignment_log10_diagnostic |
| --- | --- | --- | --- | --- |
| E1 final checkpoint by Sigma and loss | M_trace | 10 | -0.7804 | -0.6713 |
| E1 final checkpoint by Sigma and loss | M_mu | 10 | 0.3479 | 0.3479 |
| E1 final checkpoint by Sigma and loss | T | 10 | 0.7750 | 0.7024 |
| E2 spectrum scaling by panel, Sigma and d/n | M_trace | 56 | -0.6425 | -0.5710 |
| E2 spectrum scaling by panel, Sigma and d/n | M_mu | 56 | 0.1916 | -0.0459 |
| E2 spectrum scaling by panel, Sigma and d/n | T | 56 | 0.9934 | 0.9776 |
| E2 spectrum scaling by panel, Sigma and d/n | M_iso | 14 | -0.8124 | -0.9204 |
| E3 polynomial signal threshold by alpha and ||mu||_2 | M_trace | 21 | -0.5653 | -0.4891 |
| E3 polynomial signal threshold by alpha and ||mu||_2 | M_mu | 21 | 0.4750 | 0.5488 |
| E3 polynomial signal threshold by alpha and ||mu||_2 | T | 21 | 0.4029 | 0.6686 |
| E4 mu eigendirection by direction | M_trace | 3 | -0.8425 | -0.8425 |
| E4 mu eigendirection by direction | M_mu | 3 | -0.4291 | -0.7358 |
| E4 mu eigendirection by direction | T | 3 | -0.8201 | -0.8367 |

## 结果图

- `analysis/retained_experiment_summary.png`：四个保留主实验的核心结果汇总。
- `analysis/theorem_diagnostic_contrasts.png`：每个保留对照中 alignment 与 `M_trace/M_mu/T` 的并列图。
- `analysis/theorem_diagnostic_correlations.png`：alignment 与 log10 diagnostics 的相关性热图。
PDF 版本使用相同文件名后缀 `.pdf`。

## 设计结论

- `loss_comparison_anisotropic` 保留优化路径和 loss baseline；其中 isotropic panel 吸收旧 isotropic-only 实验。
- `exact_spectrum_scaling_alignment` 合并 isotropic varying-d 与旧 polynomial Panel A；固定 $n$ 变化 $d$ 已改为固定 $d=20000$ 变化 $n$。
- `exact_polynomial_spectrum_alignment` 只保留 polynomial signal threshold。
- `exact_mu_eigendirection_ablation` 保留各向异性几何消融；SVM-LS 等价检查由 `A_svm_ls` 内嵌完成。

所有数值均直接从当前保留的 `outputs/**/*.csv` 重新计算。
