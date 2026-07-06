#!/usr/bin/env python3
"""Retained exact finite-sample hard-margin SVM direction experiments."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation_utils import (
    alignment_beta_beta,
    build_design,
    ensure_dir,
    ls_solution_beta,
    make_mu,
    make_spectrum,
    plot_grouped_curve,
    signal_strength,
    svm_result_row,
    svm_solve,
    write_csv,
)


OUT_DIR = Path("outputs/svm_alignment")
MAX_D = 20000
FIXED_MU_NORM = signal_strength(MAX_D)
DIMS = [2000, 4000, 8000, 12000, 16000, 20000]
N_VALUES_FIXED_D = [50, 100, 200, 400, 800, 1200, 2000, 3000]
POLY_ALPHAS = [0.2, 0.5, 0.8]


def spectrum_specs(d: int, *, mu_direction_override: Optional[str] = None) -> List[Dict]:
    specs = [
        {
            "sigma_label": "isotropic",
            "spectrum": make_spectrum(d, "isotropic"),
            "mu_direction": mu_direction_override or "geometric",
        }
    ]
    for alpha in POLY_ALPHAS:
        specs.append(
            {
                "sigma_label": f"polynomial alpha={alpha}",
                "spectrum": make_spectrum(d, "polynomial", alpha=alpha),
                "mu_direction": mu_direction_override or "largest_eigenvector",
            }
        )
    return specs


def metric_long(df: pd.DataFrame, metrics: Sequence[str], x_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for metric in metrics:
            rows.append(
                {
                    x_col: row[x_col],
                    "seed": row.get("seed"),
                    "metric": metric,
                    "value": row.get(metric),
                    "panel": row.get("panel", "main"),
                    "alpha_label": row.get("alpha_label", row.get("alpha")),
                    "mu_norm_label": row.get("mu_norm_label", row.get("mu_norm_l2")),
                    "mu_direction": row.get("mu_direction"),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_curve(
    df: pd.DataFrame,
    *,
    x_col: str,
    metrics: Sequence[str],
    output_base: Path,
    title: str,
    xlabel: str,
    order: Sequence,
    group_order: Optional[Sequence[str]] = None,
) -> None:
    long = metric_long(df, metrics, x_col)
    if group_order is None:
        group_order = list(metrics)
    plot_grouped_curve(
        long,
        x_col=x_col,
        y_col="value",
        group_col="metric",
        output_base=output_base,
        title=title,
        xlabel=xlabel,
        ylabel="Sigma-metric direction alignment",
        order=order,
        group_order=group_order,
    )


def solve_one(
    *,
    experiment: str,
    d: int,
    n: int,
    spectrum,
    mu: np.ndarray,
    mu_direction: str,
    seed: int,
    extras: Optional[Dict] = None,
) -> Dict:
    print(f"[{experiment}] d={d} n={n} seed={seed} sigma={spectrum.sigma_type}", flush=True)
    design = build_design(n, mu, spectrum, seed)
    beta_ls = ls_solution_beta(design.K)
    svm = svm_solve(design.K)
    row = svm_result_row(
        experiment=experiment,
        seed=seed,
        n=n,
        design=design,
        beta_ls=beta_ls,
        svm=svm,
        mu_direction=mu_direction,
    )
    if extras:
        row.update(extras)
    del design, beta_ls, svm
    gc.collect()
    return row


def run_config_grid(
    *,
    experiment: str,
    configs: Sequence[Dict],
    m: int,
    seed_base: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for ci, cfg in enumerate(configs):
        for rep in range(m):
            seed = int(cfg.get("seed_base", seed_base + ci * 100000) + rep)
            row = solve_one(
                experiment=experiment,
                d=cfg["d"],
                n=cfg["n"],
                spectrum=cfg["spectrum"],
                mu=cfg["mu"],
                mu_direction=cfg["mu_direction"],
                seed=seed,
                extras=cfg.get("extras", {}),
            )
            rows.append(row)
    return rows


def run_spectrum_scaling_alignment(
    *,
    experiment: str = "exact_spectrum_scaling_alignment",
    output_name: str = "exact_spectrum_scaling_alignment",
    mu_direction_override: Optional[str] = None,
) -> pd.DataFrame:
    m = 20
    configs = []

    for d in DIMS:
        n = int(round(0.05 * d))
        for spec_index, spec in enumerate(spectrum_specs(d, mu_direction_override=mu_direction_override)):
            spectrum = spec["spectrum"]
            mu = make_mu(
                d,
                spectrum,
                norm_l2=FIXED_MU_NORM,
                direction=spec["mu_direction"],
            )
            configs.append(
                {
                    "d": d,
                    "n": n,
                    "spectrum": spectrum,
                    "mu": mu,
                    "mu_direction": spec["mu_direction"],
                    "extras": {
                        "panel": "fixed_rate_vary_d",
                        "scaling_regime": "fixed-rate",
                        "rho": 0.05,
                        "sigma_label": spec["sigma_label"],
                        "fixed_mu_norm_l2": FIXED_MU_NORM,
                    },
                    "seed_base": 3500000 + spec_index * 1000000 + d * 10,
                }
            )

    d = MAX_D
    for n in N_VALUES_FIXED_D:
        for spec_index, spec in enumerate(spectrum_specs(d, mu_direction_override=mu_direction_override)):
            spectrum = spec["spectrum"]
            mu = make_mu(
                d,
                spectrum,
                norm_l2=FIXED_MU_NORM,
                direction=spec["mu_direction"],
            )
            configs.append(
                {
                    "d": d,
                    "n": n,
                    "spectrum": spectrum,
                    "mu": mu,
                    "mu_direction": spec["mu_direction"],
                    "extras": {
                        "panel": "fixed_d_vary_n",
                        "scaling_regime": "fixed-d",
                        "sigma_label": spec["sigma_label"],
                        "fixed_mu_norm_l2": FIXED_MU_NORM,
                    },
                    "seed_base": 3560000 + spec_index * 1000000 + n * 100,
                }
            )

    rows = run_config_grid(experiment=experiment, configs=configs, m=m, seed_base=3500000)
    out = write_csv(rows, OUT_DIR / f"{output_name}.csv")
    plot_spectrum_scaling(out, output_name=output_name, mu_direction_override=mu_direction_override)
    return out


def plot_spectrum_scaling(
    df: pd.DataFrame,
    *,
    output_name: str = "exact_spectrum_scaling_alignment",
    mu_direction_override: Optional[str] = None,
) -> None:
    output_base = OUT_DIR / output_name
    ensure_dir(str(output_base) + ".png")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharey=True)
    specs = [
        ("fixed_rate_vary_d", "d", DIMS, "A_svm_wstar", "Fixed rate: A(SVM,W*) vs d"),
        ("fixed_rate_vary_d", "d", DIMS, "A_svm_ls", "Fixed rate: A(SVM,LS) vs d"),
        ("fixed_d_vary_n", "n", N_VALUES_FIXED_D, "A_svm_wstar", "Fixed d=20000: A(SVM,W*) vs n"),
        ("fixed_d_vary_n", "n", N_VALUES_FIXED_D, "A_svm_ls", "Fixed d=20000: A(SVM,LS) vs n"),
    ]
    labels = ["isotropic"] + [f"polynomial alpha={alpha}" for alpha in POLY_ALPHAS]
    markers = ["o", "s", "^", "D"]
    for ax, (panel, x_col, order, metric, title) in zip(axes.flat, specs):
        sub = df[df["panel"] == panel]
        for sigma_label, marker in zip(labels, markers):
            means, stds = [], []
            for x in order:
                vals = pd.to_numeric(
                    sub[(sub["sigma_label"] == sigma_label) & (np.isclose(pd.to_numeric(sub[x_col]), x))][metric],
                    errors="coerce",
                ).dropna()
                ax.scatter([x] * len(vals), vals, s=12, alpha=0.20)
                means.append(float(vals.mean()))
                stds.append(float(vals.std(ddof=0)))
            means_arr = np.asarray(means)
            stds_arr = np.asarray(stds)
            ax.plot(order, means_arr, marker=marker, label=sigma_label.replace("polynomial ", "poly "), linewidth=1.7)
            ax.fill_between(order, means_arr - stds_arr, means_arr + stds_arr, alpha=0.10)
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel("Sigma-metric direction alignment")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    title_suffix = "" if mu_direction_override is None else f", mu_direction={mu_direction_override}"
    fig.suptitle(f"Exact SVM spectrum scaling: ||mu||_2={FIXED_MU_NORM:.3f}{title_suffix}")
    fig.tight_layout()
    fig.savefig(str(output_base) + ".png", dpi=350)
    fig.savefig(str(output_base) + ".pdf")
    plt.close(fig)


def run_polynomial_spectrum() -> pd.DataFrame:
    experiment = "exact_polynomial_spectrum_alignment"
    mu_norms = [1, 2, 4, 6, 8, 10, 12]
    m = 20
    configs = []
    for alpha in POLY_ALPHAS:
        for mu_norm in mu_norms:
            d = MAX_D
            spectrum = make_spectrum(d, "polynomial", alpha=alpha)
            mu = make_mu(d, spectrum, norm_l2=float(mu_norm), direction="largest_eigenvector")
            configs.append(
                {
                    "d": d,
                    "n": 100,
                    "spectrum": spectrum,
                    "mu": mu,
                    "mu_direction": "largest_eigenvector",
                    "extras": {
                        "panel": "B_fixed_d_vary_mu",
                        "alpha_label": f"alpha={alpha}",
                        "mu_norm_label": str(mu_norm),
                    },
                    "seed_base": 3650000 + int(alpha * 1000) * 10000 + int(mu_norm) * 1000,
                }
    )
    rows = run_config_grid(experiment=experiment, configs=configs, m=m, seed_base=3600000)
    out = write_csv(rows, OUT_DIR / "exact_polynomial_spectrum_alignment.csv")
    plot_polynomial(out, mu_norms)
    return out


def plot_polynomial(df: pd.DataFrame, mu_norms: Sequence[int]) -> None:
    output_base = OUT_DIR / "exact_polynomial_spectrum_alignment"
    ensure_dir(str(output_base) + ".png")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    plot_specs = [
        ("B_fixed_d_vary_mu", "mu_norm_l2", mu_norms, "A_svm_wstar", "Signal threshold: A(SVM,W*) vs ||mu||_2"),
        ("B_fixed_d_vary_mu", "mu_norm_l2", mu_norms, "A_svm_ls", "Signal threshold: A(SVM,LS) vs ||mu||_2"),
    ]
    for ax, (panel, x_col, order, metric, title) in zip(axes, plot_specs):
        sub = df[df["panel"] == panel]
        for alpha_label, marker in zip([f"alpha={alpha}" for alpha in POLY_ALPHAS], ["o", "s", "^"]):
            means, stds = [], []
            for x in order:
                vals = pd.to_numeric(
                    sub[(sub["alpha_label"] == alpha_label) & (np.isclose(pd.to_numeric(sub[x_col]), x))][metric],
                    errors="coerce",
                ).dropna()
                ax.scatter([x] * len(vals), vals, s=12, alpha=0.23)
                means.append(float(vals.mean()))
                stds.append(float(vals.std(ddof=0)))
            means_arr = np.asarray(means)
            stds_arr = np.asarray(stds)
            ax.plot(order, means_arr, marker=marker, label=alpha_label, linewidth=1.7)
            ax.fill_between(order, means_arr - stds_arr, means_arr + stds_arr, alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel("Sigma-metric direction alignment")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Exact finite-sample SVM polynomial signal threshold: d=20000, n=100")
    fig.tight_layout()
    fig.savefig(str(output_base) + ".png", dpi=350)
    fig.savefig(str(output_base) + ".pdf")
    plt.close(fig)


def run_mu_eigendirection() -> pd.DataFrame:
    experiment = "exact_mu_eigendirection_ablation"
    d = 20000
    n = 100
    m = 50
    alpha = 0.8
    spectrum = make_spectrum(d, "polynomial", alpha=alpha)
    configs = []
    for direction in ["largest_eigenvector", "smallest_eigenvector", "random"]:
        for rep in range(m):
            seed = 3700000 + rep
            mu_seed = 990000 + rep if direction == "random" else 0
            mu = make_mu(d, spectrum, norm_l2=signal_strength(d), direction=direction, seed=mu_seed)
            configs.append(
                {
                    "d": d,
                    "n": n,
                    "spectrum": spectrum,
                    "mu": mu,
                    "mu_direction": direction,
                    "extras": {"paired_seed": seed},
                    "seed_base": seed,
                }
            )
    rows = run_config_grid(experiment=experiment, configs=configs, m=1, seed_base=3700000)
    out = write_csv(rows, OUT_DIR / "exact_mu_eigendirection_ablation.csv")
    plot_metric_curve(
        out,
        x_col="mu_direction",
        metrics=["A_svm_wstar", "A_svm_ls"],
        output_base=OUT_DIR / "exact_mu_eigendirection_ablation",
        title=f"Exact SVM mu eigendirection ablation: polynomial alpha=0.8, d=20000, n=100, ||mu||_2={signal_strength(d):.3f}",
        xlabel="mu direction relative to Sigma eigenvectors",
        order=["largest_eigenvector", "smallest_eigenvector", "random"],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default="all",
        choices=[
            "all",
            "exact_spectrum_scaling_alignment",
            "exact_spectrum_scaling_alignment_geometric_mu",
            "exact_polynomial_spectrum_alignment",
            "exact_mu_eigendirection_ablation",
        ],
    )
    args = parser.parse_args()
    if args.experiment == "all":
        run_spectrum_scaling_alignment()
        run_polynomial_spectrum()
        run_mu_eigendirection()
    elif args.experiment == "exact_spectrum_scaling_alignment":
        run_spectrum_scaling_alignment()
    elif args.experiment == "exact_spectrum_scaling_alignment_geometric_mu":
        run_spectrum_scaling_alignment(
            experiment="exact_spectrum_scaling_alignment_geometric_mu",
            output_name="exact_spectrum_scaling_alignment_geometric_mu",
            mu_direction_override="geometric",
        )
    elif args.experiment == "exact_polynomial_spectrum_alignment":
        run_polynomial_spectrum()
    elif args.experiment == "exact_mu_eigendirection_ablation":
        run_mu_eigendirection()


if __name__ == "__main__":
    main()
