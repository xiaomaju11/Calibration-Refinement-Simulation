#!/usr/bin/env python3
"""Analyze the retained high-value simulation experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("analysis")
DIAGNOSTIC_COLS = ["M_trace", "M_mu", "T"]
SPECTRUM_ORDER = [
    "isotropic",
    "polynomial alpha=0.2",
    "polynomial alpha=0.5",
    "polynomial alpha=0.8",
]
CONTEXT_LABELS = {
    "E1 final checkpoint by Sigma and loss": "E1 loss x Sigma",
    "E2 spectrum scaling by panel, Sigma and d/n": "E2 scaling",
    "E3 polynomial signal threshold by alpha and ||mu||_2": "E3 signal",
    "E4 mu eigendirection by direction": "E4 mu direction",
}


def mean_std_count(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df[columns].iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(rows)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def scalar(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    return float(df.loc[mask, col].iloc[0])


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(str(output_base) + ".png", dpi=350, bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def diagnostic_summary(
    df: pd.DataFrame,
    *,
    experiment: str,
    context: str,
    group_cols: list[str],
    alignment_col: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_cols = extra_cols or []
    metric_cols = [alignment_col] + DIAGNOSTIC_COLS + extra_cols
    metric_cols = [col for col in metric_cols if col in df.columns]
    out = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .rename(columns={alignment_col: "alignment_mean"})
    )
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="count")
    out = counts.merge(out, on=group_cols, how="left")
    out.insert(0, "context", context)
    out.insert(0, "experiment", experiment)
    out["alignment_metric"] = alignment_col
    return out


def range_label(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    return f"{fmt(float(vals.min()))}..{fmt(float(vals.max()))}"


def diagnostic_range_row(df: pd.DataFrame, context: str) -> dict[str, str | int]:
    return {
        "context": context,
        "groups": int(len(df)),
        "alignment_range": range_label(df["alignment_mean"]),
        "M_trace_range": range_label(df["M_trace"]),
        "M_mu_range": range_label(df["M_mu"]),
        "T_range": range_label(df["T"]),
    }


def diagnostic_correlations(df: pd.DataFrame, context: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df["alignment_mean"], errors="coerce")
    diagnostics = DIAGNOSTIC_COLS + (["M_iso"] if "M_iso" in df.columns else [])
    for diagnostic in diagnostics:
        if diagnostic not in df.columns:
            continue
        x = pd.to_numeric(df[diagnostic], errors="coerce")
        valid = x.notna() & y.notna()
        x = x[valid]
        yy = y[valid]
        if len(x) < 3 or x.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
            corr = np.nan
            log_corr = np.nan
        else:
            corr = float(x.corr(yy))
            log_corr = float(pd.Series(np.log10(x.to_numpy())).corr(yy.reset_index(drop=True))) if (x > 0).all() else np.nan
        rows.append(
            {
                "context": context,
                "diagnostic": diagnostic,
                "groups": int(len(x)),
                "corr_alignment": corr,
                "corr_alignment_log10_diagnostic": log_corr,
            }
        )
    return pd.DataFrame(rows)


def plot_curve_from_summary(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    title: str,
    xlabel: str,
    ylabel: str = "A(SVM, W*)",
) -> None:
    markers = ["o", "s", "^", "D"]
    colors = ["#4C78A8", "#F28E2B", "#59A14F", "#B07AA1"]
    for sigma_label, marker, color in zip(SPECTRUM_ORDER, markers, colors):
        sub = df[df["sigma_label"] == sigma_label].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(
            sub[x_col],
            sub["mean"],
            marker=marker,
            color=color,
            label=sigma_label.replace("polynomial ", "poly "),
            linewidth=1.7,
        )
        ax.fill_between(
            sub[x_col],
            sub["mean"] - sub["std"].fillna(0.0),
            sub["mean"] + sub["std"].fillna(0.0),
            color=color,
            alpha=0.12,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def plot_retained_experiment_summary(
    *,
    loss_delta: pd.DataFrame,
    scaling_summary: pd.DataFrame,
    poly_signal: pd.DataFrame,
    mu_dir: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.2))

    ax = axes[0, 0]
    loss_order = [
        "isotropic",
        "low_rank",
        "polynomial alpha=0.2",
        "polynomial alpha=0.5",
        "polynomial alpha=0.8",
    ]
    loss_plot = loss_delta.set_index("sigma_label").loc[loss_order].reset_index()
    x = np.arange(len(loss_plot))
    ax.bar(x, loss_plot["mean"], yerr=loss_plot["std"], color="#4C78A8", alpha=0.85, capsize=3)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["iso", "low-rank", "poly .2", "poly .5", "poly .8"])
    ax.set_title("E1 square - logistic final alignment")
    ax.set_ylabel("Delta A(w_t, W*) at t=500")
    ax.grid(True, axis="y", alpha=0.25)

    fixed_rate = scaling_summary[scaling_summary["panel"] == "fixed_rate_vary_d"]
    plot_curve_from_summary(
        axes[0, 1],
        fixed_rate,
        x_col="d",
        title="E2 fixed-rate scaling",
        xlabel="dimension d",
    )

    fixed_d = scaling_summary[scaling_summary["panel"] == "fixed_d_vary_n"]
    plot_curve_from_summary(
        axes[0, 2],
        fixed_d,
        x_col="n",
        title="E2 fixed d=20000 sample-size scaling",
        xlabel="sample size n",
    )

    ax = axes[1, 0]
    for alpha, marker, color in [
        (0.2, "o", "#B07AA1"),
        (0.5, "s", "#F28E2B"),
        (0.8, "^", "#76B7B2"),
    ]:
        sub = poly_signal[np.isclose(poly_signal["alpha"], alpha)].sort_values("mu_norm_l2")
        ax.plot(sub["mu_norm_l2"], sub["mean"], marker=marker, color=color, label=f"alpha={alpha}", linewidth=1.8)
        ax.fill_between(
            sub["mu_norm_l2"],
            sub["mean"] - sub["std"].fillna(0.0),
            sub["mean"] + sub["std"].fillna(0.0),
            color=color,
            alpha=0.14,
        )
    ax.set_title("E3 signal threshold")
    ax.set_xlabel("||mu||_2")
    ax.set_ylabel("A(SVM, W*)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    direction_order = ["largest_eigenvector", "smallest_eigenvector", "random"]
    mu_plot = mu_dir.set_index("mu_direction").loc[direction_order].reset_index()
    x = np.arange(len(mu_plot))
    ax.bar(x, mu_plot["mean"], yerr=mu_plot["std"], color="#9C755F", alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["largest", "smallest", "random"])
    ax.set_title("E4 mu eigendirection")
    ax.set_ylabel("A(SVM, W*)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    axes[1, 2].axis("off")
    fig.suptitle("Retained experiment result summary")
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "retained_experiment_summary")


def diagnostic_plot_label(row: pd.Series) -> str:
    context = str(row["context"])
    if context.startswith("E1"):
        return f"{row['sigma_label']}\n{row['method']}".replace("polynomial alpha=", "poly ")
    if context.startswith("E2"):
        sigma = str(row["sigma_label"]).replace("polynomial alpha=", "poly ")
        if row["panel"] == "fixed_rate_vary_d":
            return f"rate\n{sigma}\nd={int(row['d'])}"
        return f"d fixed\n{sigma}\nn={int(row['n'])}"
    if context.startswith("E3"):
        return f"a={row['alpha']:g}\nmu={row['mu_norm_l2']:g}"
    if context.startswith("E4"):
        return str(row["mu_direction"]).replace("_eigenvector", "")
    return str(row.name)


def sort_diagnostic_context(df: pd.DataFrame) -> pd.DataFrame:
    context = str(df["context"].iloc[0])
    out = df.copy()
    if context.startswith("E1"):
        sigma_order = {
            "isotropic": 0,
            "low_rank": 1,
            "polynomial alpha=0.2": 2,
            "polynomial alpha=0.5": 3,
            "polynomial alpha=0.8": 4,
        }
        method_order = {"logistic": 0, "square": 1}
        out["_sigma_order"] = out["sigma_label"].map(sigma_order)
        out["_method_order"] = out["method"].map(method_order)
        out = out.sort_values(["_sigma_order", "_method_order"])
    elif context.startswith("E2"):
        panel_order = {"fixed_rate_vary_d": 0, "fixed_d_vary_n": 1}
        sigma_order = {label: i for i, label in enumerate(SPECTRUM_ORDER)}
        out["_panel_order"] = out["panel"].map(panel_order)
        out["_sigma_order"] = out["sigma_label"].map(sigma_order)
        out = out.sort_values(["_panel_order", "_sigma_order", "d", "n"])
    elif context.startswith("E3"):
        out = out.sort_values(["alpha", "mu_norm_l2"])
    elif context.startswith("E4"):
        direction_order = {"largest_eigenvector": 0, "smallest_eigenvector": 1, "random": 2}
        out["_direction_order"] = out["mu_direction"].map(direction_order)
        out = out.sort_values("_direction_order")
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def plot_theorem_diagnostic_contrasts(theorem_diagnostics: pd.DataFrame) -> None:
    contexts = list(CONTEXT_LABELS)
    fig, axes = plt.subplots(len(contexts), 1, figsize=(20.0, 16.5))
    diagnostic_colors = {
        "M_trace": "#4C78A8",
        "M_mu": "#F28E2B",
        "T": "#59A14F",
        "M_iso": "#B07AA1",
    }
    for ax, context in zip(axes, contexts):
        sub = sort_diagnostic_context(theorem_diagnostics[theorem_diagnostics["context"] == context])
        x = np.arange(len(sub))
        labels = [diagnostic_plot_label(row) for _, row in sub.iterrows()]
        ax.bar(x, sub["alignment_mean"], color="#BAB0AC", alpha=0.8, label="alignment mean")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("alignment")
        ax.set_title(CONTEXT_LABELS.get(context, context))
        ax.grid(True, axis="y", alpha=0.24)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)

        ax2 = ax.twinx()
        diagnostics = DIAGNOSTIC_COLS + (["M_iso"] if "M_iso" in sub.columns and sub["M_iso"].notna().any() else [])
        for diagnostic in diagnostics:
            vals = pd.to_numeric(sub[diagnostic], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            vals = vals.where(vals > 0)
            ax2.plot(
                x,
                np.log10(vals),
                marker="o",
                markersize=2.8,
                linewidth=1.0,
                color=diagnostic_colors.get(diagnostic),
                label=f"log10 {diagnostic}",
            )
        ax2.set_ylabel("log10 diagnostic")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right", fontsize=8)

    fig.suptitle("Diagnostics contrast: alignment versus theory diagnostics")
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "theorem_diagnostic_contrasts")


def plot_theorem_diagnostic_correlations(diagnostic_corr: pd.DataFrame) -> None:
    corr = diagnostic_corr.copy()
    corr["context_label"] = corr["context"].map(CONTEXT_LABELS).fillna(corr["context"])
    contexts = [CONTEXT_LABELS[c] for c in CONTEXT_LABELS]
    diagnostics = ["M_trace", "M_mu", "T", "M_iso"]
    pivot = corr.pivot(index="context_label", columns="diagnostic", values="corr_alignment_log10_diagnostic")
    pivot = pivot.reindex(index=contexts, columns=diagnostics)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(diagnostics)))
    ax.set_xticklabels(diagnostics)
    ax.set_yticks(np.arange(len(contexts)))
    ax.set_yticklabels(contexts)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt(float(data[i, j])), ha="center", va="center", fontsize=9)
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=9, color="#555555")
    ax.set_title("Correlation between alignment and log10 diagnostics")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "theorem_diagnostic_correlations")


def build_analysis() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    loss = load_csv("outputs/loss_comparison/loss_comparison_anisotropic.csv")
    loss_final = loss[loss["step"] == 500].copy()
    loss_means = mean_std_count(loss_final, ["sigma_label", "method"], "alignment_to_wstar")
    loss_wide = loss_final.pivot_table(index=["sigma_label", "seed"], columns="method", values="alignment_to_wstar")
    loss_wide["square_minus_logistic"] = loss_wide["square"] - loss_wide["logistic"]
    loss_delta = (
        loss_wide.groupby("sigma_label")["square_minus_logistic"]
        .agg(mean="mean", std="std", min="min", max="max", count="count")
        .reset_index()
    )

    scaling = load_csv("outputs/svm_alignment/exact_spectrum_scaling_alignment.csv")
    scaling_summary = mean_std_count(scaling, ["panel", "sigma_label", "d", "n"], "A_svm_wstar")
    scaling_ls_summary = mean_std_count(scaling, ["panel", "sigma_label", "d", "n"], "A_svm_ls")
    scaling_rate = scaling_summary[scaling_summary["panel"] == "fixed_rate_vary_d"].copy()
    scaling_fixed_d = scaling_summary[scaling_summary["panel"] == "fixed_d_vary_n"].copy()
    scaling_sample_range = (
        scaling_fixed_d.groupby("sigma_label")["mean"]
        .agg(min="min", max="max")
        .assign(effect_range=lambda x: x["max"] - x["min"])
        .reset_index()
    )
    scaling_rate_at_max_d = scaling_rate[scaling_rate["d"] == 20000].copy()

    poly = load_csv("outputs/svm_alignment/exact_polynomial_spectrum_alignment.csv")
    poly_b = poly[poly["panel"] == "B_fixed_d_vary_mu"].copy()
    poly_signal = mean_std_count(poly_b, ["alpha", "mu_norm_l2"], "A_svm_wstar")
    signal_range = (
        poly_signal.groupby("alpha")["mean"]
        .agg(min="min", max="max")
        .assign(effect_range=lambda x: x["max"] - x["min"])
        .reset_index()
    )

    mu = load_csv("outputs/svm_alignment/exact_mu_eigendirection_ablation.csv")
    mu_dir = mean_std_count(mu, ["mu_direction"], "A_svm_wstar")
    mu_ls = mean_std_count(mu, ["mu_direction"], "A_svm_ls")
    mu_sigma = mean_std_count(mu, ["mu_direction"], "mu_norm_sigma")
    mu_sigma_inv = mean_std_count(mu, ["mu_direction"], "mu_norm_sigma_inv")
    pair_col = "paired_seed" if "paired_seed" in mu.columns else "seed"
    mu_wide = mu.pivot_table(index=pair_col, columns="mu_direction", values="A_svm_wstar")
    mu_pair_rows = []
    for left, right in [
        ("smallest_eigenvector", "largest_eigenvector"),
        ("random", "largest_eigenvector"),
        ("smallest_eigenvector", "random"),
    ]:
        delta = mu_wide[left] - mu_wide[right]
        mu_pair_rows.append(
            {
                "contrast": f"{left} - {right}",
                "mean": delta.mean(),
                "std": delta.std(),
                "min": delta.min(),
                "max": delta.max(),
                "count": delta.count(),
            }
        )
    mu_pair = pd.DataFrame(mu_pair_rows)

    loss_diag = diagnostic_summary(
        loss_final,
        experiment="loss_comparison_anisotropic",
        context="E1 final checkpoint by Sigma and loss",
        group_cols=["sigma_label", "method"],
        alignment_col="alignment_to_wstar",
        extra_cols=["mu_norm_sigma", "mu_norm_sigma_inv"],
    )
    scaling_diag = diagnostic_summary(
        scaling,
        experiment="exact_spectrum_scaling_alignment",
        context="E2 spectrum scaling by panel, Sigma and d/n",
        group_cols=["panel", "sigma_label", "d", "n"],
        alignment_col="A_svm_wstar",
        extra_cols=["M_iso", "mu_norm_l2", "mu_norm_sigma", "mu_norm_sigma_inv"],
    )
    poly_diag = diagnostic_summary(
        poly_b,
        experiment="exact_polynomial_spectrum_alignment",
        context="E3 polynomial signal threshold by alpha and ||mu||_2",
        group_cols=["alpha", "mu_norm_l2"],
        alignment_col="A_svm_wstar",
        extra_cols=["mu_norm_sigma", "mu_norm_sigma_inv"],
    )
    mu_diag = diagnostic_summary(
        mu,
        experiment="exact_mu_eigendirection_ablation",
        context="E4 mu eigendirection by direction",
        group_cols=["mu_direction"],
        alignment_col="A_svm_wstar",
        extra_cols=["mu_norm_sigma", "mu_norm_sigma_inv"],
    )
    diagnostic_tables = [loss_diag, scaling_diag, poly_diag, mu_diag]
    theorem_diagnostics = pd.concat(diagnostic_tables, ignore_index=True, sort=False)
    diagnostic_ranges = pd.DataFrame([diagnostic_range_row(df, df["context"].iloc[0]) for df in diagnostic_tables])
    diagnostic_corr = pd.concat(
        [diagnostic_correlations(df, df["context"].iloc[0]) for df in diagnostic_tables],
        ignore_index=True,
        sort=False,
    )

    candidates = [
        {
            "rank": 1,
            "experiment": "exact_polynomial_spectrum_alignment",
            "contrast": "Polynomial spectrum signal-strength threshold",
            "main_metric": "A_svm_wstar",
            "effect_summary": "range by alpha: alpha=0.2 "
            + fmt(scalar(signal_range, signal_range["alpha"] == 0.2, "effect_range"))
            + "; alpha=0.5 "
            + fmt(scalar(signal_range, signal_range["alpha"] == 0.5, "effect_range"))
            + "; alpha=0.8 "
            + fmt(scalar(signal_range, signal_range["alpha"] == 0.8, "effect_range")),
            "comparability": "Strong within each alpha: d=20000, n=100, mu direction fixed; only ||mu||_2 changes.",
            "retained_reason": "Most theory-aligned signal-threshold contrast.",
        },
        {
            "rank": 2,
            "experiment": "exact_mu_eigendirection_ablation",
            "contrast": "Mu eigendirection ablation",
            "main_metric": "A_svm_wstar and A_svm_ls",
            "effect_summary": "A_svm_wstar means: largest "
            + fmt(scalar(mu_dir, mu_dir["mu_direction"] == "largest_eigenvector", "mean"))
            + ", smallest "
            + fmt(scalar(mu_dir, mu_dir["mu_direction"] == "smallest_eigenvector", "mean"))
            + ", random "
            + fmt(scalar(mu_dir, mu_dir["mu_direction"] == "random", "mean")),
            "comparability": "Very strong paired design: same d,n,Sigma,||mu||_2 and design seed; only mu direction changes.",
            "retained_reason": "Cleanest anisotropic geometry ablation.",
        },
        {
            "rank": 3,
            "experiment": "exact_spectrum_scaling_alignment",
            "contrast": "Spectrum scaling: fixed-rate d-sweep and fixed-d n-sweep",
            "main_metric": "A_svm_wstar",
            "effect_summary": "fixed-d sample-size ranges from "
            + fmt(scaling_sample_range["effect_range"].min())
            + " to "
            + fmt(scaling_sample_range["effect_range"].max())
            + "; fixed-rate d=20000 means span "
            + fmt(scaling_rate_at_max_d["mean"].min())
            + " to "
            + fmt(scaling_rate_at_max_d["mean"].max()),
            "comparability": "Fixed-rate panel compares spectra along n/d=0.05; fixed-d panel compares n at d=20000. Non-identity Sigma uses ||mu||_2 fixed at 2sqrt(log 20000).",
            "retained_reason": "Replaces the weak fixed-n varying-d design and adds polynomial spectra to fixed-rate scaling.",
        },
        {
            "rank": 4,
            "experiment": "loss_comparison_anisotropic",
            "contrast": "Loss function x covariance spectrum",
            "main_metric": "final-step A(w_t,W*) and paired square-logistic delta",
            "effect_summary": "square-logistic final delta ranges from "
            + fmt(loss_delta["mean"].min())
            + " to "
            + fmt(loss_delta["mean"].max())
            + "; polynomial alpha=0.8 delta "
            + fmt(scalar(loss_delta, loss_delta["sigma_label"] == "polynomial alpha=0.8", "mean")),
            "comparability": "Very strong within each Sigma: same seed, data and checkpoints; only loss update changes.",
            "retained_reason": "Only retained algorithmic baseline for optimization-bias effects.",
        },
    ]
    ranking = pd.DataFrame(candidates)

    comparability = pd.DataFrame(
        [
            {
                "experiment_or_contrast": "loss_comparison_anisotropic",
                "intended_factor": "loss function within each Sigma; Sigma family across panels",
                "controlled_variables": "within each sigma_label: d=20000; n=1000; mu_direction=largest_eigenvector; mu_norm_l2; seed; design matrix; checkpoints",
                "pairing": "paired logistic vs square by seed and design",
                "main_metric": "alignment_to_wstar; alignment_logistic_square; square-logistic paired delta",
                "verdict": "retain as E1",
                "absorbed_old_experiments": "loss_comparison_isotropic; logistic GD sample-size path sanity checks",
                "caveat": "cross-Sigma comparisons also change mu_norm_sigma and theorem diagnostics",
            },
            {
                "experiment_or_contrast": "exact_spectrum_scaling_alignment",
                "intended_factor": "fixed-rate d-scaling and fixed-d sample-size scaling across spectra",
                "controlled_variables": "fixed-rate: n/d=0.05; fixed-d: d=20000; non-identity Sigma uses ||mu||_2=2sqrt(log 20000)",
                "pairing": "matched by d grid in fixed-rate panel and n grid in fixed-d panel",
                "main_metric": "A_svm_wstar; A_svm_ls; A_ls_wstar; M_trace; M_mu; T",
                "verdict": "retain as E2",
                "absorbed_old_experiments": "exact_isotropic_alignment_varying_d; polynomial Panel A fixed-n varying-d",
                "caveat": "cross-spectrum comparisons intentionally change Sigma spectrum and mu Sigma-norm diagnostics",
            },
            {
                "experiment_or_contrast": "exact_polynomial_spectrum_alignment",
                "intended_factor": "mu_norm_l2 within each alpha",
                "controlled_variables": "d=20000; n=100; mu_direction=largest_eigenvector; exact SVM solver",
                "pairing": "not paired across mu_norm_l2",
                "main_metric": "A_svm_wstar; A_svm_ls; spectrum diagnostics",
                "verdict": "retain as E3",
                "absorbed_old_experiments": "exact_theorem_alignment_trends signal-strength component",
                "caveat": "cross-alpha comparison intentionally changes Sigma spectrum",
            },
            {
                "experiment_or_contrast": "exact_mu_eigendirection_ablation",
                "intended_factor": "mu direction relative to Sigma eigenvectors",
                "controlled_variables": "d=20000; n=100; alpha=0.8; Sigma; mu_norm_l2; exact SVM solver; design seed",
                "pairing": "paired by design seed across directions",
                "main_metric": "A_svm_wstar; A_svm_ls; mu_norm_sigma; mu_norm_sigma_inv",
                "verdict": "retain as E4",
                "absorbed_old_experiments": "standalone SVM-LS equivalence diagnostic",
                "caveat": "random direction adds a controlled extra random mu draw",
            },
        ]
    )

    group_summary_parts = []
    for name, df in [
        ("loss_final_alignment", loss_means),
        ("loss_paired_delta", loss_delta),
        ("scaling_alignment", scaling_summary),
        ("scaling_svm_ls", scaling_ls_summary),
        ("scaling_fixed_d_sample_range", scaling_sample_range),
        ("scaling_fixed_rate_at_d20000", scaling_rate_at_max_d),
        ("poly_signal", poly_signal),
        ("poly_signal_range", signal_range),
        ("mu_direction_alignment", mu_dir),
        ("mu_direction_svm_ls", mu_ls),
        ("mu_direction_pair_delta", mu_pair),
        ("mu_norm_sigma", mu_sigma),
        ("mu_norm_sigma_inv", mu_sigma_inv),
        ("theorem_diagnostic_ranges", diagnostic_ranges),
        ("theorem_diagnostic_correlations", diagnostic_corr),
    ]:
        part = df.copy()
        part.insert(0, "table", name)
        group_summary_parts.append(part)
    group_summary = pd.concat(group_summary_parts, ignore_index=True, sort=False)

    ranking.to_csv(OUT_DIR / "candidate_contrast_ranking.csv", index=False)
    group_summary.to_csv(OUT_DIR / "group_summary_tables.csv", index=False)
    comparability.to_csv(OUT_DIR / "comparability_matrix.csv", index=False)
    theorem_diagnostics.to_csv(OUT_DIR / "theorem_diagnostic_contrasts.csv", index=False)
    diagnostic_corr.to_csv(OUT_DIR / "theorem_diagnostic_correlations.csv", index=False)
    plot_retained_experiment_summary(
        loss_delta=loss_delta,
        scaling_summary=scaling_summary,
        poly_signal=poly_signal,
        mu_dir=mu_dir,
    )
    plot_theorem_diagnostic_contrasts(theorem_diagnostics)
    plot_theorem_diagnostic_correlations(diagnostic_corr)

    report = []
    report.append("# 精简实验设计分析报告")
    report.append("")
    report.append("## 保留的四个主实验")
    report.append("")
    report.append(markdown_table(ranking, ["rank", "experiment", "contrast", "effect_summary", "comparability", "retained_reason"]))
    report.append("")
    report.append("## 变量可比较性审计")
    report.append("")
    report.append(markdown_table(comparability, ["experiment_or_contrast", "intended_factor", "controlled_variables", "pairing", "absorbed_old_experiments", "caveat"]))
    report.append("")
    report.append("## 关键数值证据")
    report.append("")

    loss_delta_print = loss_delta.copy()
    for col in ["mean", "std", "min", "max"]:
        loss_delta_print[col] = loss_delta_print[col].map(fmt)
    report.append("### Loss function paired deltas")
    report.append("")
    report.append(markdown_table(loss_delta_print, ["sigma_label", "mean", "std", "min", "max", "count"]))
    report.append("")

    scaling_print = scaling_sample_range.copy()
    for col in ["min", "max", "effect_range"]:
        scaling_print[col] = scaling_print[col].map(fmt)
    report.append("### Fixed-d sample-size ranges")
    report.append("")
    report.append(markdown_table(scaling_print, ["sigma_label", "min", "max", "effect_range"]))
    report.append("")

    rate_print = scaling_rate_at_max_d[["sigma_label", "d", "n", "mean", "std", "count"]].copy()
    for col in ["mean", "std"]:
        rate_print[col] = rate_print[col].map(fmt)
    report.append("### Fixed-rate panel at d=20000")
    report.append("")
    report.append(markdown_table(rate_print, ["sigma_label", "d", "n", "mean", "std", "count"]))
    report.append("")

    signal_print = signal_range.copy()
    for col in ["min", "max", "effect_range"]:
        signal_print[col] = signal_print[col].map(fmt)
    report.append("### Polynomial signal-strength ranges")
    report.append("")
    report.append(markdown_table(signal_print, ["alpha", "min", "max", "effect_range"]))
    report.append("")

    mu_print = mu_dir.copy()
    for col in ["mean", "std"]:
        mu_print[col] = mu_print[col].map(fmt)
    report.append("### Mu eigendirection means")
    report.append("")
    report.append(markdown_table(mu_print, ["mu_direction", "mean", "std", "count"]))
    report.append("")

    report.append("## 理论诊断量对比")
    report.append("")
    report.append("Diagnostics contrast 不是新的数据生成实验，而是对四个保留实验的 CSV 做二次汇总：按每个实验的核心对照维度分组，同时列出方向对齐均值与 `M_trace`、`M_mu`、`T` 等理论诊断量，用来判断观察到的方向差异是否伴随理论控制量变化。")
    report.append("")
    report.append("### Diagnostic ranges by retained contrast")
    report.append("")
    report.append(markdown_table(diagnostic_ranges, ["context", "groups", "alignment_range", "M_trace_range", "M_mu_range", "T_range"]))
    report.append("")
    corr_print = diagnostic_corr.copy()
    for col in ["corr_alignment", "corr_alignment_log10_diagnostic"]:
        corr_print[col] = corr_print[col].map(fmt)
    report.append("### Alignment correlations with diagnostics")
    report.append("")
    report.append(markdown_table(corr_print, ["context", "diagnostic", "groups", "corr_alignment", "corr_alignment_log10_diagnostic"]))
    report.append("")

    report.append("## 结果图")
    report.append("")
    report.append("- `analysis/retained_experiment_summary.png`：四个保留主实验的核心结果汇总。")
    report.append("- `analysis/theorem_diagnostic_contrasts.png`：每个保留对照中 alignment 与 `M_trace/M_mu/T` 的并列图。")
    report.append("- `analysis/theorem_diagnostic_correlations.png`：alignment 与 log10 diagnostics 的相关性热图。")
    report.append("PDF 版本使用相同文件名后缀 `.pdf`。")
    report.append("")

    report.append("## 设计结论")
    report.append("")
    report.append("- `loss_comparison_anisotropic` 保留优化路径和 loss baseline；其中 isotropic panel 吸收旧 isotropic-only 实验。")
    report.append("- `exact_spectrum_scaling_alignment` 合并 isotropic varying-d 与旧 polynomial Panel A；固定 $n$ 变化 $d$ 已改为固定 $d=20000$ 变化 $n$。")
    report.append("- `exact_polynomial_spectrum_alignment` 只保留 polynomial signal threshold。")
    report.append("- `exact_mu_eigendirection_ablation` 保留各向异性几何消融；SVM-LS 等价检查由 `A_svm_ls` 内嵌完成。")
    report.append("")
    report.append("所有数值均直接从当前保留的 `outputs/**/*.csv` 重新计算。")
    report.append("")
    (OUT_DIR / "experiment_comparison_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    build_analysis()
