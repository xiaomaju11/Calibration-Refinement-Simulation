#!/usr/bin/env python3
"""Retained logistic-loss versus square-loss GD baseline experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from simulation_utils import (
    CHECKPOINTS,
    alignment_beta_beta,
    build_design,
    gd_rows_for_states,
    make_mu,
    make_spectrum,
    plot_grouped_curve,
    run_gd_paths,
    signal_strength,
    write_csv,
)


OUT_DIR = Path("outputs/loss_comparison")


Config = Tuple[str, Optional[float]]


def sigma_label(sigma_type: str, alpha: Optional[float]) -> str:
    if sigma_type == "polynomial":
        return f"polynomial alpha={alpha}"
    return sigma_type


def run_configs(experiment: str, configs: List[Config]) -> None:
    d = 20000
    n = d // 20
    m = 100
    rows = []
    for sigma_type, alpha in configs:
        spectrum = make_spectrum(d, sigma_type, alpha=alpha)
        mu = make_mu(d, spectrum, norm_l2=signal_strength(d), direction="largest_eigenvector")
        label = sigma_label(sigma_type, alpha)
        for rep in range(m):
            seed = 2100000 + rep + int(1000 * (0 if alpha is None else alpha)) + 10000 * configs.index((sigma_type, alpha))
            print(f"[{experiment}] sigma={label} seed={seed}", flush=True)
            design = build_design(n, mu, spectrum, seed)
            paths = run_gd_paths(design, t_max=500, checkpoints=CHECKPOINTS, logistic=True, square=True)
            pair_alignment: Dict[int, float] = {}
            for step in CHECKPOINTS:
                pair_alignment[step] = alignment_beta_beta(
                    paths["logistic"][step],
                    paths["square"][step],
                    design.H,
                )
            for method in ["logistic", "square"]:
                eta = paths["_eta"][method]  # type: ignore[index]
                seed_rows = gd_rows_for_states(
                    experiment=experiment,
                    method=method,
                    seed=seed,
                    n=n,
                    design=design,
                    states=paths[method],
                    beta_ls=None,
                    mu_direction="largest_eigenvector",
                    eta=eta,
                )
                for row in seed_rows:
                    row["sigma_label"] = label
                    row["sigma_method"] = f"{label} / {method}"
                    row["alignment_logistic_square"] = pair_alignment[int(row["step"])]
                    row["loss_baseline_note"] = "algorithmic baseline comparison, not theorem verification"
                rows.extend(seed_rows)

    if experiment != "loss_comparison_anisotropic":
        raise ValueError(f"unknown retained experiment: {experiment}")
    output_name = experiment
    df = write_csv(rows, OUT_DIR / f"{output_name}.csv")
    group_order = []
    for sigma_type, alpha in configs:
        label = sigma_label(sigma_type, alpha)
        group_order.extend([f"{label} / logistic", f"{label} / square"])
    plot_grouped_curve(
        df,
        x_col="step",
        y_col="alignment_to_wstar",
        group_col="sigma_method",
        output_base=OUT_DIR / output_name,
        title=(
            "Loss baseline comparison, not theorem verification: "
            f"d=20000, n=1000, ||mu||_2={signal_strength(d):.3f}, mu || v_max(Sigma)"
        ),
        xlabel="GD checkpoint t",
        ylabel="A(w_t, W*) in Sigma metric",
        order=CHECKPOINTS,
        group_order=group_order,
        figsize=(11.5, 6.2),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "loss_comparison_anisotropic"],
    )
    args = parser.parse_args()
    if args.experiment in ("all", "loss_comparison_anisotropic"):
        run_configs(
            "loss_comparison_anisotropic",
            [
                ("isotropic", None),
                ("low_rank", None),
                ("polynomial", 0.2),
                ("polynomial", 0.5),
                ("polynomial", 0.8),
            ],
        )


if __name__ == "__main__":
    main()
