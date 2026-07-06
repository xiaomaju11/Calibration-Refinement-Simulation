# Binary Direction Simulation

This project simulates direction learning in a binary Gaussian-mixture model.
It compares logistic-gradient-descent directions, square-loss-gradient-descent
directions, exact hard-margin SVM directions, least-squares directions, and the
population reference direction

$$
W^*=2\Sigma^{-1}\mu .
$$

The current design keeps four main experiments. Older experiments with highly
overlapping scientific content are not run as standalone experiments anymore;
their checks are absorbed into retained CSV diagnostics, figures, and the
analysis report.

## Notation

### Data Model

For sample $i$,

$$
Y_i\in\{-1,+1\},\qquad
X_i=Y_i\mu+Z_i,\qquad
Z_i\sim N(0,\Sigma).
$$

| Symbol | Definition |
| --- | --- |
| $i$ | sample index, $i=1,\ldots,n$ |
| $d$ | feature dimension |
| $n$ | training sample size |
| $m$ | number of Monte Carlo replications |
| $Y_i$ | binary label for sample $i$ |
| $X_i\in\mathbb{R}^d$ | observed feature vector |
| $Z_i$ | Gaussian noise |
| $\mu\in\mathbb{R}^d$ | class-mean signal direction |
| $\Sigma\in\mathbb{R}^{d\times d}$ | positive-definite covariance matrix |
| $\lambda_j$ | the $j$-th eigenvalue of $\Sigma$; $\Sigma$ is diagonal in this project |
| $\alpha$ | polynomial-spectrum decay exponent |
| $W^*$ | reference direction, $W^*=2\Sigma^{-1}\mu$ |

The code solves all finite-sample problems through the signed design

$$
S_i=Y_iX_i,\qquad S\in\mathbb{R}^{n\times d},\qquad K=SS^\top,
$$

where $K$ is the exact finite-sample Gram matrix.

### Directions and Alignment

All direction alignments use the corresponding $\Sigma$-metric:

$$
A_\Sigma(w_1,w_2)=
\frac{w_1^\top\Sigma w_2}
{\sqrt{w_1^\top\Sigma w_1}\sqrt{w_2^\top\Sigma w_2}}.
$$

| Symbol | Definition | CSV field |
| --- | --- | --- |
| $w_t^{\mathrm{log}}$ | logistic-loss GD direction at checkpoint $t$ | `method=logistic` |
| $w_t^{\mathrm{sq}}$ | square-loss GD direction at checkpoint $t$ | `method=square` |
| $\widehat w_{\mathrm{SVM}}$ | exact hard-margin SVM direction | `A_svm_*` fields |
| $\widehat w_{\mathrm{LS}}$ | least-squares direction | `A_*_ls` fields |
| $A_\Sigma(w_t,W^*)$ | GD direction alignment with $W^*$ | `alignment_to_wstar` |
| $A_\Sigma(w_t,\widehat w_{\mathrm{LS}})$ | GD direction alignment with LS | `alignment_to_ls` |
| $A_\Sigma(\widehat w_{\mathrm{SVM}},W^*)$ | SVM direction alignment with $W^*$ | `A_svm_wstar` |
| $A_\Sigma(\widehat w_{\mathrm{SVM}},\widehat w_{\mathrm{LS}})$ | SVM direction alignment with LS | `A_svm_ls` |
| $A_\Sigma(\widehat w_{\mathrm{LS}},W^*)$ | LS direction alignment with $W^*$ | `A_ls_wstar` |

### Norms and Theoretical Diagnostics

The $\Sigma$- and $\Sigma^{-1}$-norms are

$$
\lVert v\rVert_\Sigma=\sqrt{v^\top\Sigma v},
\qquad
\lVert v\rVert_{\Sigma^{-1}}=\sqrt{v^\top\Sigma^{-1}v}.
$$

Spectrum quantities are

$$
\mathrm{tr}(\Sigma)=\sum_{j=1}^d\lambda_j,\qquad
\lVert\Sigma\rVert_2=\max_j\lambda_j,\qquad
\lVert\Sigma\rVert_F=\sqrt{\sum_{j=1}^d\lambda_j^2}.
$$

The diagnostic denominator is

$$
D(n,\mu,\Sigma)=
\max\left\{
n^{3/2}\lVert\Sigma\rVert_2,\,
n\lVert\Sigma\rVert_F,\,
n\sqrt{\log n}\lVert\mu\rVert_\Sigma
\right\}.
$$

The main diagnostics are

$$
M_{\mathrm{trace}}
=\frac{\mathrm{tr}(\Sigma)}{D(n,\mu,\Sigma)},
\qquad
M_\mu
=\frac{\lVert\mu\rVert_2^2}{\lVert\mu\rVert_\Sigma},
$$

$$
T
=
\frac{n\lVert\mu\rVert_2^4}
{n\lVert\mu\rVert_\Sigma^2+\lVert\Sigma\rVert_F^2+n\lVert\Sigma\rVert_2^2}.
$$

For the isotropic case $\Sigma=I_d$, the code also records

$$
M_{\mathrm{iso}}
=
\frac{d}
{\max\{n^2,\;n\sqrt{\log n}\lVert\mu\rVert_2\}}.
$$

| CSV field | Mathematical symbol | Expression |
| --- | --- | --- |
| `mu_norm_l2` | $\lVert\mu\rVert_2$ | $\sqrt{\sum_{j=1}^d\mu_j^2}$ |
| `mu_norm_sigma` | $\lVert\mu\rVert_\Sigma$ | $\sqrt{\mu^\top\Sigma\mu}$ |
| `mu_norm_sigma_inv` | $\lVert\mu\rVert_{\Sigma^{-1}}$ | $\sqrt{\mu^\top\Sigma^{-1}\mu}$ |
| `trace_sigma` | $\mathrm{tr}(\Sigma)$ | $\sum_{j=1}^d\lambda_j$ |
| `sigma_op_norm` | $\lVert\Sigma\rVert_2$ | $\max_{1\le j\le d}\lambda_j$ |
| `sigma_fro_norm` | $\lVert\Sigma\rVert_F$ | $\sqrt{\sum_{j=1}^d\lambda_j^2}$ |
| `M_trace` | $M_{\mathrm{trace}}$ | $\mathrm{tr}(\Sigma)/D(n,\mu,\Sigma)$ |
| `M_mu` | $M_\mu$ | $\lVert\mu\rVert_2^2/\lVert\mu\rVert_\Sigma$ |
| `T` | $T$ | $\frac{n\lVert\mu\rVert_2^4}{n\lVert\mu\rVert_\Sigma^2+\lVert\Sigma\rVert_F^2+n\lVert\Sigma\rVert_2^2}$ |
| `M_iso` | $M_{\mathrm{iso}}$ | $\frac{d}{\max\{n^2,\;n\sqrt{\log n}\lVert\mu\rVert_2\}}$ |

## Core Rules

1. Do not report test accuracy, test error, ROC, AUC, or other test-risk metrics.
2. Do not shrink main experiments into smoke runs. The $d$, $n$, $m$, parameter grids, and checkpoints below define the full main-pass scale.
3. SVM experiments must use the real finite-sample Gram matrix $K=SS^\top$, not the deterministic equivalent

$$
Q_n=\mathrm{tr}(\Sigma)I_n+\lVert\mu\rVert_2^2\mathbf{1}\mathbf{1}^\top.
$$

4. All direction metrics must use $A_\Sigma(\cdot,\cdot)$.
5. CSV outputs must keep theory diagnostics and solver diagnostics so that SVM-LS behavior, spectrum effects, signal strength, and finite-sample effects can be audited after the run.

## Covariance Structures

### `isotropic`

$$
\Sigma=I_d.
$$

This is the clean baseline and the high-dimensional scaling check associated
with Corollary-3.3-type behavior.

### `low_rank`

The `low_rank non-identity Sigma` setting is not a low-rank covariance matrix
and is not singular. In the code it is a full-rank positive-definite matrix:

$$
\Sigma=\mathrm{diag}(\lambda_1,\ldots,\lambda_d),
$$

where almost all $\lambda_j=1$, and only 16 spike coordinates have
$\lambda_j=2$. It should therefore be read as

$$
\Sigma=I_d+\Delta,\qquad
\mathrm{rank}(\Delta)=16.
$$

This structure is used only as a non-identity covariance control in the loss
baseline, not as an independent main SVM experiment.

### `polynomial`

$$
\lambda_k=k^{-\alpha},\qquad
\alpha\in\{0.2,0.5,0.8\}.
$$

This is the main anisotropic spectrum family. It is used to study spectrum
decay, signal-strength thresholds, and the effect of the signal direction
relative to $\Sigma$'s eigenvectors.

## Shared Implementation Requirements

The default signal strength is

$$
s_\mu(d)=2\sqrt{\log d},
$$

unless an experiment explicitly scans $\lVert\mu\rVert_2$. The GD checkpoint
set is fixed as

$$
\mathcal{T}_{\mathrm{GD}}
=\{0,10,20,\ldots,100,200,300,400,500\}.
$$

Other requirements:

- hard-margin SVM uses an exact finite-sample dual solve;
- least squares uses a linear solve, not an explicit matrix inverse;
- every output experiment saves `.csv`, `.png`, and `.pdf`;
- figures must not imply test risk, and should only show direction alignment and diagnostics.

## Retained Main Experiments

### E1. `loss_comparison_anisotropic`

Run:

```bash
python3 simulate_identical_logistic_gd_alignment.py --experiment loss_comparison_anisotropic
```

Goal: compare $w_t^{\mathrm{log}}$ and $w_t^{\mathrm{sq}}$ on exactly the
same training data and checkpoint set $\mathcal{T}_{\mathrm{GD}}$, and test
whether different $\Sigma$ structures amplify optimization-path differences.

Full scale:

- $d=20000$.
- $n=d/20=1000$.
- $m=100$.
- $t\in\mathcal{T}_{\mathrm{GD}}$.
- For every $\Sigma$, $\mu\parallel v_{\max}(\Sigma)$ and $\lVert\mu\rVert_2=s_\mu(20000)\approx6.294$. When $\Sigma=I_d$, the top eigenspace is non-unique, and the code uses one coordinate direction.
- $\Sigma$ families:
  - `isotropic`: $\Sigma=I_d$.
  - `low_rank`: $\Sigma=I_d+\Delta$, $\mathrm{rank}(\Delta)=16$.
  - `polynomial, alpha=0.2`: $\lambda_k=k^{-0.2}$.
  - `polynomial, alpha=0.5`: $\lambda_k=k^{-0.5}$.
  - `polynomial, alpha=0.8`: $\lambda_k=k^{-0.8}$.

Why retained:

- It absorbs the old `loss_comparison_isotropic`, because the $\Sigma=I_d$ curve is now one panel in this experiment.
- It absorbs the old logistic-GD sample-size path sanity check without keeping a weak standalone GD sweep.
- It is the only retained algorithmic baseline for loss-function and covariance-spectrum interactions.

Required checks:

- Within the same `sigma_label`, $w_t^{\mathrm{log}}$ and $w_t^{\mathrm{sq}}$ must share the same seed, design matrix $S$, $\mu\parallel v_{\max}(\Sigma)$, and checkpoint set.
- The CSV must save

$$
A_\Sigma(w_t^{\mathrm{log}},w_t^{\mathrm{sq}})
$$

as `alignment_logistic_square`.
- This experiment is only an optimization-bias baseline, not an SVM theorem verification.

Outputs:

```text
outputs/loss_comparison/loss_comparison_anisotropic.csv
outputs/loss_comparison/loss_comparison_anisotropic.png
outputs/loss_comparison/loss_comparison_anisotropic.pdf
```

### E2. `exact_spectrum_scaling_alignment`

Run:

```bash
python3 simulate_svm_alignment.py --experiment exact_spectrum_scaling_alignment
```

Goal: merge the old `exact_isotropic_alignment_varying_d` experiment and the
old polynomial Panel A. This experiment no longer uses "fixed $n$, varying
$d$" as a main contrast. Instead it studies:

1. fixed-rate high-dimensional scaling with $\rho=n/d=0.05$ and varying $d$;
2. fixed-$d$ sample-size scaling with $d=20000$ and varying $n$.

The fixed-rate panel includes both $\Sigma=I_d$ and polynomial spectra. For
non-identity $\Sigma$,

$$
\lVert\mu\rVert_2=s_\mu(20000)=2\sqrt{\log 20000}\approx6.294.
$$

Full scale:

Fixed-rate panel:

$$
d\in\{2000,4000,8000,12000,16000,20000\}.
$$

$$
\rho=\frac{n}{d}=0.05,\qquad n=0.05d.
$$

Fixed-$d$ panel:

$$
d=20000,\qquad
n\in\{50,100,200,400,800,1200,2000,3000\}.
$$

Both panels compare

$$
\Sigma\in
\left\{
I_d,\;
\mathrm{diag}(1^{-\alpha},2^{-\alpha},\ldots,d^{-\alpha})
\;:\;\alpha\in\{0.2,0.5,0.8\}
\right\}.
$$

Other settings:

- $m=20$.
- $\Sigma=I_d$ uses a geometric $\mu$ direction.
- polynomial spectra use $\mu\parallel v_{\max}(\Sigma)$.
- all non-identity $\Sigma$ settings fix $\lVert\mu\rVert_2\approx6.294$.

Why retained:

- It adds polynomial spectra to fixed-rate scaling, avoiding a scaling analysis only under $\Sigma=I_d$.
- The old polynomial Panel A with fixed $n=100$ and varying $d$ is not kept as a standalone experiment because it mixes dimension growth with sample scarcity; it is replaced by varying $n$ at fixed $d=20000$.
- It compares isotropic and non-identity spectra in the same experiment, clarifying how spectrum shape affects $A_\Sigma(\widehat w_{\mathrm{SVM}},W^*)$.

Required checks:

- `panel` must distinguish `fixed_rate_vary_d` and `fixed_d_vary_n`.
- The fixed-rate panel must include $\Sigma=I_d$ and polynomial spectra with $\alpha\in\{0.2,0.5,0.8\}$.
- The fixed-$d$ panel must fix $d=20000$ and vary only $n$.
- Non-identity $\Sigma$ settings must record and verify $\lVert\mu\rVert_2\approx6.294$.
- The CSV must save $A_\Sigma(\widehat w_{\mathrm{SVM}},W^*)$, $A_\Sigma(\widehat w_{\mathrm{SVM}},\widehat w_{\mathrm{LS}})$, $A_\Sigma(\widehat w_{\mathrm{LS}},W^*)$, $M_{\mathrm{trace}}$, $M_\mu$, $T$, $M_{\mathrm{iso}}$, and solver diagnostics.

Outputs:

```text
outputs/svm_alignment/exact_spectrum_scaling_alignment.csv
outputs/svm_alignment/exact_spectrum_scaling_alignment.png
outputs/svm_alignment/exact_spectrum_scaling_alignment.pdf
```

Additional rerun:

```bash
python3 simulate_svm_alignment.py --experiment exact_spectrum_scaling_alignment_geometric_mu
```

This rerun keeps the same full scale, the same $d/n$ grid, and the same
$\lVert\mu\rVert_2\approx6.294$, but uses a geometric $\mu$ direction for
all spectra. It does not overwrite the main E2 outputs; it writes separate
files:

```text
outputs/svm_alignment/exact_spectrum_scaling_alignment_geometric_mu.csv
outputs/svm_alignment/exact_spectrum_scaling_alignment_geometric_mu.png
outputs/svm_alignment/exact_spectrum_scaling_alignment_geometric_mu.pdf
```

### E3. `exact_polynomial_spectrum_alignment`

Run:

```bash
python3 simulate_svm_alignment.py --experiment exact_polynomial_spectrum_alignment
```

Goal: study how signal strength $\lVert\mu\rVert_2$ affects
$\widehat w_{\mathrm{SVM}}$ under polynomial spectra.

Full scale:

$$
\alpha\in\{0.2,0.5,0.8\}.
$$

With $d=20000$ and $n=100$, scan

$$
\lVert\mu\rVert_2\in\{1,2,4,6,8,10,12\}.
$$

Other settings:

- $m=20$.
- $\mu$ is aligned with the largest-eigenvalue direction of $\Sigma$.

Why retained:

- It is the cleanest signal-threshold main experiment.
- The old Panel A was merged into E2; this experiment only keeps the clean contrast that varies $\lVert\mu\rVert_2$ at fixed $d,n$.
- It absorbs the core $\alpha$- and $\lVert\mu\rVert_2$-related information from the old `exact_theorem_alignment_trends`; the old trend figure is no longer generated as a standalone simulation.

Required checks:

- Within each $\alpha$, only $\lVert\mu\rVert_2$ may change.
- Cross-$\alpha$ comparisons must be interpreted as changing the spectrum structure, not only signal strength.
- The CSV must save $\mathrm{tr}(\Sigma)$, $\lVert\Sigma\rVert_F$, $\lVert\Sigma\rVert_2$, $\lVert\mu\rVert_\Sigma$, $\lVert\mu\rVert_{\Sigma^{-1}}$, $M_{\mathrm{trace}}$, $M_\mu$, and $T$.

Outputs:

```text
outputs/svm_alignment/exact_polynomial_spectrum_alignment.csv
outputs/svm_alignment/exact_polynomial_spectrum_alignment.png
outputs/svm_alignment/exact_polynomial_spectrum_alignment.pdf
```

### E4. `exact_mu_eigendirection_ablation`

Run:

```bash
python3 simulate_svm_alignment.py --experiment exact_mu_eigendirection_ablation
```

Goal: fix $\Sigma$, $d$, $n$, and $\lVert\mu\rVert_2$, and vary only the
direction of $\mu$ relative to $\Sigma$'s eigenvectors.

Full scale:

- $d=20000$.
- $n=100$.
- $\Sigma$ is polynomial with $\alpha=0.8$, i.e. $\lambda_k=k^{-0.8}$.
- $m=50$.
- $\mu$ directions:
  - `largest_eigenvector`: $\mu\parallel v_{\max}(\Sigma)$.
  - `smallest_eigenvector`: $\mu\parallel v_{\min}(\Sigma)$.
  - `random`: $\mu/\lVert\mu\rVert_2$ is a random unit direction.

Why retained:

- It is the cleanest anisotropic-geometry ablation.
- It directly tests how the relationship between $\mu$ and $\Sigma$'s eigenvectors affects $W^*=2\Sigma^{-1}\mu$ and $\widehat w_{\mathrm{SVM}}$.

Required checks:

- The three directions must share the same $d$, $n$, $\Sigma$, and $\lVert\mu\rVert_2$.
- The three directions are paired by `paired_seed`; the `random` direction may use an additional controlled `mu_seed`.
- The CSV must save $\lVert\mu\rVert_\Sigma$, $\lVert\mu\rVert_{\Sigma^{-1}}$, $A_\Sigma(\widehat w_{\mathrm{SVM}},W^*)$, $A_\Sigma(\widehat w_{\mathrm{SVM}},\widehat w_{\mathrm{LS}})$, and solver diagnostics.

Outputs:

```text
outputs/svm_alignment/exact_mu_eigendirection_ablation.csv
outputs/svm_alignment/exact_mu_eigendirection_ablation.png
outputs/svm_alignment/exact_mu_eigendirection_ablation.pdf
```

## Merged or Downgraded Older Experiments

| Old experiment | Treatment | Where the check is retained |
| --- | --- | --- |
| `gd_alignment_by_sample_size` | not a main experiment | E1 retains the $\{w_t^{\mathrm{log}}:t\in\mathcal{T}_{\mathrm{GD}}\}$ optimization path |
| `gd_alignment_by_ls` | not a main experiment | E1 and SVM CSV files retain $A_\Sigma(\cdot,\widehat w_{\mathrm{LS}})$ diagnostics |
| `loss_comparison_isotropic` | merged | $\Sigma=I_d$ panel in E1 |
| `exact_svm_alignment_fixed_d` | old entry and outputs removed | E2-E4 keep exact finite-sample SVM directions and solver diagnostics |
| `exact_svm_alignment_fixed_rate` | merged | `fixed_rate_vary_d` panel in E2 |
| `exact_svm_ls_equivalence_fixed_d` | no standalone run | E2-E4 include $A_\Sigma(\widehat w_{\mathrm{SVM}},\widehat w_{\mathrm{LS}})$ and $A_\Sigma(\widehat w_{\mathrm{LS}},W^*)$ |
| `exact_isotropic_alignment_fixed_d` | merged | `fixed_d_vary_n` panel in E2 |
| old polynomial Panel A: fixed $n$, varying $d$ | merged and rewritten | fixed-rate and fixed-$d$ sample-size panels in E2 |
| `exact_theorem_alignment_trends` | analysis report only | `analysis/experiment_comparison_report.md` |

## Diagnostics Contrast

`diagnostics contrast` is not a new simulation experiment and does not change
any $d$, $n$, $m$, or grid size. It is an analysis-layer comparison: it
reads the CSV files from the retained experiments, groups rows by each
experiment's core contrast variables, and displays mean alignment next to the
theoretical diagnostics.

The diagnostic fields are:

| CSV field | Mathematical symbol | Expression | Scope |
| --- | --- | --- | --- |
| `M_trace` | $M_{\mathrm{trace}}$ | $\mathrm{tr}(\Sigma)/D(n,\mu,\Sigma)$ | all main experiments |
| `M_mu` | $M_\mu$ | $\lVert\mu\rVert_2^2/\lVert\mu\rVert_\Sigma$ | all main experiments |
| `T` | $T$ | $\frac{n\lVert\mu\rVert_2^4}{n\lVert\mu\rVert_\Sigma^2+\lVert\Sigma\rVert_F^2+n\lVert\Sigma\rVert_2^2}$ | all main experiments |
| `M_iso` | $M_{\mathrm{iso}}$ | $\frac{d}{\max\{n^2,\;n\sqrt{\log n}\lVert\mu\rVert_2\}}$ | $\Sigma=I_d$ only |

Grouping:

- E1: final checkpoint $t=500$, grouped by `sigma_label x method`. Within each `sigma_label`, $w_t^{\mathrm{log}}$ and $w_t^{\mathrm{sq}}$ share the same $M_{\mathrm{trace}}$, $M_\mu$, and $T$, so diagnostics contrast separates loss-driven direction differences from diagnostic-driven differences.
- E2: grouped by `panel x sigma_label x d x n`. `fixed_rate_vary_d` compares spectra along $\rho=n/d=0.05$; `fixed_d_vary_n` compares sample-size effects at $d=20000$.
- E3: grouped by $\alpha\times\lVert\mu\rVert_2$, for checking signal threshold behavior against $M_\mu$ and $T$.
- E4: grouped by `mu_direction`, for checking how changing only the signal eigendirection changes $\lVert\mu\rVert_\Sigma$, $\lVert\mu\rVert_{\Sigma^{-1}}$, $M_\mu$, $T$, and alignment.

Additional analysis outputs:

```text
analysis/theorem_diagnostic_contrasts.csv
analysis/theorem_diagnostic_correlations.csv
analysis/theorem_diagnostic_contrasts.png
analysis/theorem_diagnostic_contrasts.pdf
analysis/theorem_diagnostic_correlations.png
analysis/theorem_diagnostic_correlations.pdf
```

## Running Experiments

Run all retained experiments:

```bash
python3 simulate_identical_logistic_gd_alignment.py --experiment all
python3 simulate_svm_alignment.py --experiment all
python3 analyze_experiment_contrasts.py
```

Run an individual SVM main experiment or the E2 geometric-$\mu$ rerun:

```bash
python3 simulate_svm_alignment.py --experiment exact_spectrum_scaling_alignment
python3 simulate_svm_alignment.py --experiment exact_spectrum_scaling_alignment_geometric_mu
python3 simulate_svm_alignment.py --experiment exact_polynomial_spectrum_alignment
python3 simulate_svm_alignment.py --experiment exact_mu_eigendirection_ablation
```

## Included Output Files

```text
outputs/loss_comparison/loss_comparison_anisotropic.csv
outputs/loss_comparison/loss_comparison_anisotropic.png
outputs/loss_comparison/loss_comparison_anisotropic.pdf
outputs/svm_alignment/exact_spectrum_scaling_alignment.csv
outputs/svm_alignment/exact_spectrum_scaling_alignment.png
outputs/svm_alignment/exact_spectrum_scaling_alignment.pdf
outputs/svm_alignment/exact_spectrum_scaling_alignment_geometric_mu.csv
outputs/svm_alignment/exact_spectrum_scaling_alignment_geometric_mu.png
outputs/svm_alignment/exact_spectrum_scaling_alignment_geometric_mu.pdf
outputs/svm_alignment/exact_polynomial_spectrum_alignment.csv
outputs/svm_alignment/exact_polynomial_spectrum_alignment.png
outputs/svm_alignment/exact_polynomial_spectrum_alignment.pdf
outputs/svm_alignment/exact_mu_eigendirection_ablation.csv
outputs/svm_alignment/exact_mu_eigendirection_ablation.png
outputs/svm_alignment/exact_mu_eigendirection_ablation.pdf
```

The analysis script writes:

```text
analysis/candidate_contrast_ranking.csv
analysis/comparability_matrix.csv
analysis/group_summary_tables.csv
analysis/theorem_diagnostic_contrasts.csv
analysis/theorem_diagnostic_correlations.csv
analysis/retained_experiment_summary.png
analysis/retained_experiment_summary.pdf
analysis/theorem_diagnostic_contrasts.png
analysis/theorem_diagnostic_contrasts.pdf
analysis/theorem_diagnostic_correlations.png
analysis/theorem_diagnostic_correlations.pdf
analysis/experiment_comparison_report.md
```

## Completion Audit

After a full experiment cleanup:

1. `simulate_identical_logistic_gd_alignment.py --experiment all` should run only E1.
2. `simulate_svm_alignment.py --experiment all` should run only E2, E3, and E4.
3. `outputs/` should not contain artifacts from merged or downgraded old experiments.
4. `analysis/experiment_comparison_report.md` should be recalculated only from retained experiments.
5. `.DS_Store` and similar non-experiment artifacts should not be present in the output directories.
