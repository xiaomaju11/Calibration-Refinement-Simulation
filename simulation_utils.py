#!/usr/bin/env python3
"""Shared utilities for the binary Gaussian-mixture direction experiments."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh


COMMON_COLUMNS = [
    "experiment",
    "method",
    "seed",
    "d",
    "n",
    "n_over_d",
    "sigma_type",
    "alpha",
    "mu_direction",
    "mu_norm_l2",
    "mu_norm_sigma",
    "mu_norm_sigma_inv",
    "trace_sigma",
    "sigma_op_norm",
    "sigma_fro_norm",
    "M_trace",
    "M_mu",
    "T",
    "step",
    "alignment_to_wstar",
    "alignment_to_ls",
    "alignment_to_svm",
    "loss_value",
    "min_margin",
    "svm_primal_violation",
    "svm_dual_feasibility_min_alpha",
    "svm_kkt_complementarity",
    "svm_primal_dual_gap",
    "support_vector_fraction",
    "solver_status",
]


CHECKPOINTS = list(range(0, 101, 10)) + [200, 300, 400, 500]


def signal_strength(d: int) -> float:
    return 2.0 * math.sqrt(math.log(d))


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Sequence[Dict], path: str | Path) -> pd.DataFrame:
    ensure_dir(path)
    df = pd.DataFrame(rows)
    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    ordered = COMMON_COLUMNS + [c for c in df.columns if c not in COMMON_COLUMNS]
    df = df[ordered]
    df.to_csv(path, index=False, na_rep="NA")
    return df


@dataclass(frozen=True)
class Spectrum:
    sigma_type: str
    lambdas: np.ndarray
    alpha: float = math.nan
    lambda_signal: float = math.nan
    lambda_bulk: float = math.nan
    bulk_rank: int = 0
    bulk_random_fraction: float = math.nan
    spike_indices: Tuple[int, ...] = ()

    @property
    def d(self) -> int:
        return int(self.lambdas.size)

    @property
    def trace(self) -> float:
        return float(np.sum(self.lambdas))

    @property
    def op_norm(self) -> float:
        return float(np.max(self.lambdas))

    @property
    def fro_norm(self) -> float:
        return float(np.sqrt(np.sum(self.lambdas * self.lambdas)))


def make_spectrum(
    d: int,
    sigma_type: str,
    *,
    alpha: Optional[float] = None,
    lambda_signal: float = 2.0,
    lambda_bulk: float = 1.0,
    bulk_rank: int = 16,
    bulk_random_fraction: float = 0.9,
    spectrum_seed: int = 314159,
) -> Spectrum:
    if sigma_type == "isotropic":
        return Spectrum(sigma_type=sigma_type, lambdas=np.ones(d, dtype=np.float64))

    if sigma_type == "polynomial":
        if alpha is None:
            raise ValueError("alpha is required for polynomial spectrum")
        k = np.arange(1, d + 1, dtype=np.float64)
        return Spectrum(
            sigma_type=sigma_type,
            lambdas=np.power(k, -float(alpha), dtype=np.float64),
            alpha=float(alpha),
        )

    if sigma_type == "low_rank":
        lambdas = np.full(d, float(lambda_bulk), dtype=np.float64)
        rank = min(int(bulk_rank), d)
        random_count = int(round(rank * float(bulk_random_fraction)))
        random_count = min(max(random_count, 0), rank)
        deterministic_count = rank - random_count
        deterministic = list(range(deterministic_count))
        remaining = np.setdiff1d(np.arange(d), np.array(deterministic, dtype=int), assume_unique=True)
        rng = np.random.default_rng(spectrum_seed + d + rank)
        if random_count > 0:
            random_part = rng.choice(remaining, size=random_count, replace=False).tolist()
        else:
            random_part = []
        spikes = tuple(sorted(set(deterministic + random_part)))
        lambdas[list(spikes)] = float(lambda_signal)
        return Spectrum(
            sigma_type=sigma_type,
            lambdas=lambdas,
            alpha=math.nan,
            lambda_signal=float(lambda_signal),
            lambda_bulk=float(lambda_bulk),
            bulk_rank=rank,
            bulk_random_fraction=float(bulk_random_fraction),
            spike_indices=spikes,
        )

    raise ValueError(f"unknown sigma_type: {sigma_type}")


def geometric_mu(d: int, norm_l2: float, *, ratio: float = 0.9999) -> np.ndarray:
    idx = np.arange(d, dtype=np.float64)
    values = np.power(ratio, idx, dtype=np.float64)
    values *= float(norm_l2) / np.linalg.norm(values)
    return values


def nested_geometric_mus(
    dims: Sequence[int],
    final_norm_l2: Optional[float] = None,
    *,
    ratio: float = 0.9999,
) -> Dict[int, np.ndarray]:
    max_d = int(max(dims))
    if final_norm_l2 is None:
        final_norm_l2 = signal_strength(max_d)
    base = geometric_mu(max_d, float(final_norm_l2), ratio=ratio)
    return {int(d): base[: int(d)].copy() for d in dims}


def make_mu(
    d: int,
    spectrum: Spectrum,
    *,
    norm_l2: Optional[float] = None,
    norm_sigma: Optional[float] = None,
    direction: str = "geometric",
    seed: int = 0,
) -> np.ndarray:
    if norm_l2 is None and norm_sigma is None:
        norm_l2 = signal_strength(d)

    if direction == "largest_eigenvector":
        mu = np.zeros(d, dtype=np.float64)
        mu[int(np.argmax(spectrum.lambdas))] = 1.0
    elif direction == "smallest_eigenvector":
        mu = np.zeros(d, dtype=np.float64)
        mu[int(np.argmin(spectrum.lambdas))] = 1.0
    elif direction == "random":
        rng = np.random.default_rng(seed)
        mu = rng.standard_normal(d)
        norm = np.linalg.norm(mu)
        if norm == 0:
            mu[0] = 1.0
        else:
            mu /= norm
    elif direction == "geometric":
        mu = geometric_mu(d, 1.0)
    else:
        raise ValueError(f"unknown mu direction: {direction}")

    if norm_sigma is not None:
        current = math.sqrt(float(np.sum(spectrum.lambdas * mu * mu)))
        if current == 0:
            raise ValueError("cannot scale zero mu to Sigma norm")
        mu = mu * (float(norm_sigma) / current)
    else:
        current = np.linalg.norm(mu)
        if current == 0:
            raise ValueError("cannot scale zero mu to L2 norm")
        mu = mu * (float(norm_l2) / current)
    return mu


def mu_stats(mu: np.ndarray, spectrum: Spectrum) -> Dict[str, float]:
    lambdas = spectrum.lambdas
    mu2 = mu * mu
    norm_l2 = float(np.linalg.norm(mu))
    norm_sigma = float(np.sqrt(np.sum(lambdas * mu2)))
    norm_sigma_inv = float(np.sqrt(np.sum(mu2 / lambdas)))
    return {
        "mu_norm_l2": norm_l2,
        "mu_norm_sigma": norm_sigma,
        "mu_norm_sigma_inv": norm_sigma_inv,
    }


def diagnostics(n: int, mu: np.ndarray, spectrum: Spectrum) -> Dict[str, float]:
    stats = mu_stats(mu, spectrum)
    logn = max(math.log(max(n, 2)), 1e-12)
    denom = max(
        (n ** 1.5) * spectrum.op_norm,
        n * spectrum.fro_norm,
        n * math.sqrt(logn) * stats["mu_norm_sigma"],
    )
    m_trace = spectrum.trace / denom if denom > 0 else math.nan
    m_mu = (
        stats["mu_norm_l2"] ** 2 / stats["mu_norm_sigma"]
        if stats["mu_norm_sigma"] > 0
        else math.nan
    )
    t_num = n * stats["mu_norm_l2"] ** 4
    t_den = (
        n * stats["mu_norm_sigma"] ** 2
        + spectrum.fro_norm**2
        + n * spectrum.op_norm**2
    )
    theorem_t = t_num / t_den if t_den > 0 else math.nan
    return {
        **stats,
        "trace_sigma": spectrum.trace,
        "sigma_op_norm": spectrum.op_norm,
        "sigma_fro_norm": spectrum.fro_norm,
        "M_trace": m_trace,
        "M_mu": m_mu,
        "T": theorem_t,
    }


def isotropic_margin(n: int, d: int, mu_norm_l2: float) -> float:
    return d / max(n * n, n * math.sqrt(math.log(max(n, 2))) * mu_norm_l2)


@dataclass
class Design:
    S: np.ndarray
    K: np.ndarray
    H: np.ndarray
    q_mu: np.ndarray
    spectrum: Spectrum
    mu: np.ndarray
    condition_number: float


def sample_signed_design(n: int, mu: np.ndarray, spectrum: Spectrum, seed: int) -> np.ndarray:
    """Sample signed rows S_i = y_i x_i = mu + eps_i with eps_i ~ N(0, Sigma)."""
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((int(n), spectrum.d), dtype=np.float64)
    S *= np.sqrt(spectrum.lambdas, dtype=np.float64)
    S += mu
    return S


def gram_sigma_metric(S: np.ndarray, K: np.ndarray, spectrum: Spectrum) -> np.ndarray:
    if spectrum.sigma_type == "isotropic":
        return K.copy()
    if spectrum.sigma_type == "low_rank" and spectrum.lambda_bulk == 1.0:
        H = K.copy()
        for idx in spectrum.spike_indices:
            delta = spectrum.lambdas[idx] - 1.0
            if delta != 0:
                col = S[:, idx]
                H += delta * np.outer(col, col)
        return H
    weighted = S * spectrum.lambdas
    return weighted @ S.T


def build_design(n: int, mu: np.ndarray, spectrum: Spectrum, seed: int) -> Design:
    S = sample_signed_design(n, mu, spectrum, seed)
    K = S @ S.T
    K = 0.5 * (K + K.T)
    H = gram_sigma_metric(S, K, spectrum)
    H = 0.5 * (H + H.T)
    q_mu = S @ mu
    condition_number = estimate_condition_number(K)
    return Design(S=S, K=K, H=H, q_mu=q_mu, spectrum=spectrum, mu=mu, condition_number=condition_number)


def estimate_condition_number(K: np.ndarray) -> float:
    if K.shape[0] <= 2:
        vals = np.linalg.eigvalsh(K)
        return float(vals[-1] / max(vals[0], 1e-300))
    try:
        hi = float(eigsh(K, k=1, which="LA", return_eigenvectors=False, tol=1e-3, maxiter=300)[0])
        lo = float(eigsh(K, k=1, which="SA", return_eigenvectors=False, tol=1e-3, maxiter=600)[0])
        return hi / max(lo, 1e-300)
    except Exception:
        diag = np.diag(K)
        return float(np.max(diag) / max(np.min(diag), 1e-300))


def solve_spd(K: np.ndarray, b: np.ndarray) -> np.ndarray:
    jitter = 0.0
    eye = None
    for _ in range(5):
        try:
            mat = K if jitter == 0.0 else K + jitter * eye
            c, lower = cho_factor(mat, lower=True, check_finite=False)
            return cho_solve((c, lower), b, check_finite=False)
        except Exception:
            if eye is None:
                eye = np.eye(K.shape[0], dtype=K.dtype)
            scale = max(float(np.trace(K) / max(K.shape[0], 1)), 1.0)
            jitter = scale * (1e-10 if jitter == 0.0 else jitter * 10.0 / scale)
    return np.linalg.solve(K, b)


def beta_norm_K(beta: np.ndarray, K: np.ndarray) -> float:
    val = float(beta @ (K @ beta))
    return math.sqrt(max(val, 0.0))


def beta_norm_H(beta: np.ndarray, H: np.ndarray) -> float:
    val = float(beta @ (H @ beta))
    return math.sqrt(max(val, 0.0))


def alignment_beta_wstar(beta: np.ndarray, design: Design) -> float:
    norm_beta = beta_norm_H(beta, design.H)
    mu_inv = mu_stats(design.mu, design.spectrum)["mu_norm_sigma_inv"]
    if norm_beta == 0 or mu_inv == 0:
        return 0.0
    return float((beta @ design.q_mu) / (norm_beta * mu_inv))


def alignment_beta_beta(beta1: np.ndarray, beta2: np.ndarray, H: np.ndarray) -> float:
    n1 = beta_norm_H(beta1, H)
    n2 = beta_norm_H(beta2, H)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float((beta1 @ (H @ beta2)) / (n1 * n2))


def logistic_loss_from_margins(margins: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, -margins)))


def square_loss_from_margins(margins: np.ndarray) -> float:
    return float(np.mean((1.0 - margins) ** 2))


def stable_sigmoid_negative(margins: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(margins))


def estimate_largest_eigenvalue(K: np.ndarray, *, iters: int = 30) -> float:
    rng = np.random.default_rng(12345 + K.shape[0])
    v = rng.standard_normal(K.shape[0])
    v /= np.linalg.norm(v)
    val = 0.0
    for _ in range(iters):
        Kv = K @ v
        norm = np.linalg.norm(Kv)
        if norm == 0:
            return 0.0
        v = Kv / norm
        val = float(v @ (K @ v))
    return max(val, float(np.max(np.diag(K))))


def ls_solution_beta(K: np.ndarray) -> np.ndarray:
    return solve_spd(K, np.ones(K.shape[0], dtype=np.float64))


def run_gd_paths(
    design: Design,
    *,
    t_max: int = 500,
    checkpoints: Sequence[int] = CHECKPOINTS,
    logistic: bool = True,
    square: bool = False,
    eta_logistic: Optional[float] = None,
    eta_square: Optional[float] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    K = design.K
    eig_max = estimate_largest_eigenvalue(K)
    eta_l = (2.0 * K.shape[0] / eig_max) if eta_logistic is None else float(eta_logistic)
    eta_s = (0.5 * K.shape[0] / eig_max) if eta_square is None else float(eta_square)
    wanted = set(int(s) for s in checkpoints)
    out: Dict[str, Dict[int, np.ndarray]] = {}

    if logistic:
        beta = np.zeros(K.shape[0], dtype=np.float64)
        states: Dict[int, np.ndarray] = {}
        for step in range(t_max + 1):
            if step in wanted:
                states[step] = beta.copy()
            if step == t_max:
                break
            margins = K @ beta
            beta += (eta_l / K.shape[0]) * stable_sigmoid_negative(margins)
        out["logistic"] = states

    if square:
        beta = np.zeros(K.shape[0], dtype=np.float64)
        states = {}
        for step in range(t_max + 1):
            if step in wanted:
                states[step] = beta.copy()
            if step == t_max:
                break
            margins = K @ beta
            beta += (2.0 * eta_s / K.shape[0]) * (1.0 - margins)
        out["square"] = states

    out["_eta"] = {"logistic": eta_l, "square": eta_s}  # type: ignore[assignment]
    return out


def svm_solve(K: np.ndarray, *, tol: float = 1e-7) -> Dict[str, object]:
    ones = np.ones(K.shape[0], dtype=np.float64)
    beta_ls = solve_spd(K, ones)
    polished = False
    polish_iterations = 0
    if float(np.min(beta_ls)) >= -tol:
        beta = np.maximum(beta_ls, 0.0)
        margins = K @ beta
        status = "all_support_linear_kkt"
        nit = 0
        success = True
        message = "K^{-1}1 is nonnegative; exact active-set solution"
    else:
        x0 = np.maximum(beta_ls, 0.0)

        def fun(x: np.ndarray) -> Tuple[float, np.ndarray]:
            kx = K @ x
            return float(0.5 * x @ kx - np.sum(x)), kx - ones

        res = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * K.shape[0],
            options={
                "maxiter": 10000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
                "maxcor": 20,
            },
        )
        beta = np.asarray(res.x, dtype=np.float64)
        status = f"lbfgsb:{res.status}:{res.message}"
        nit = int(res.nit)
        success = bool(res.success)
        message = str(res.message)

    beta_polished, did_polish, polish_iterations = active_set_polish(K, beta, tol=1e-10)
    if did_polish:
        beta = beta_polished
        polished = True
        status = f"{status}|active_set_polished"

    margins = K @ beta
    alpha_readme = 2.0 * beta
    primal = float(beta @ margins)
    dual = float(2.0 * np.sum(beta) - beta @ margins)
    gap = primal - dual
    primal_violation = float(np.max(np.maximum(0.0, 1.0 - margins)))
    dual_feas_min = float(np.min(alpha_readme))
    kkt = float(np.max(np.abs(alpha_readme * (margins - 1.0))))
    support_fraction = float(np.mean(beta > max(1e-9, 1e-7 * max(float(np.max(beta)), 1.0))))
    return {
        "beta": beta,
        "alpha": alpha_readme,
        "solver_status": status,
        "solver_success": success,
        "solver_message": message,
        "solver_nit": nit,
        "active_set_polished": polished,
        "active_set_polish_iterations": polish_iterations,
        "svm_primal_violation": primal_violation,
        "svm_dual_feasibility_min_alpha": dual_feas_min,
        "svm_kkt_complementarity": kkt,
        "svm_primal_dual_gap": gap,
        "support_vector_fraction": support_fraction,
    }


def active_set_polish(K: np.ndarray, beta0: np.ndarray, *, tol: float = 1e-10) -> Tuple[np.ndarray, bool, int]:
    margins0 = K @ beta0
    violation0 = np.max(np.maximum(0.0, 1.0 - margins0))
    complementarity0 = np.max(np.abs(beta0 * (margins0 - 1.0)))
    if violation0 <= tol and complementarity0 <= tol:
        return beta0, False, 0

    n = K.shape[0]
    threshold = max(1e-12, 1e-8 * max(float(np.max(beta0)), 1.0))
    active = set(np.flatnonzero(beta0 > threshold).tolist())
    if not active:
        active.add(int(np.argmin(margins0)))

    beta = beta0.copy()
    max_iter = max(20, 2 * n)
    for it in range(1, max_iter + 1):
        active_idx = np.array(sorted(active), dtype=int)
        Kaa = K[np.ix_(active_idx, active_idx)]
        sol = solve_spd(Kaa, np.ones(active_idx.size, dtype=np.float64))
        nonpositive = sol <= tol
        if np.any(nonpositive) and active_idx.size > 1:
            for idx in active_idx[nonpositive]:
                active.discard(int(idx))
            continue

        beta = np.zeros(n, dtype=np.float64)
        beta[active_idx] = np.maximum(sol, 0.0)
        margins = K @ beta
        violation = 1.0 - margins
        max_violation = float(np.max(violation))
        if max_violation <= tol:
            return beta, True, it
        active.add(int(np.argmax(violation)))

    return beta, False, max_iter


def base_row(
    *,
    experiment: str,
    method: str,
    seed: int,
    n: int,
    spectrum: Spectrum,
    mu: np.ndarray,
    mu_direction: str,
    step: float | int | str = math.nan,
) -> Dict:
    d = spectrum.d
    row = {
        "experiment": experiment,
        "method": method,
        "seed": seed,
        "d": d,
        "n": n,
        "n_over_d": n / d,
        "sigma_type": spectrum.sigma_type,
        "alpha": spectrum.alpha,
        "lambda_signal": spectrum.lambda_signal,
        "lambda_bulk": spectrum.lambda_bulk,
        "bulk_rank": spectrum.bulk_rank,
        "bulk_random_fraction": spectrum.bulk_random_fraction,
        "mu_direction": mu_direction,
        "step": step,
    }
    row.update(diagnostics(n, mu, spectrum))
    if spectrum.sigma_type == "isotropic":
        row["M_iso"] = isotropic_margin(n, d, row["mu_norm_l2"])
    return row


def gd_rows_for_states(
    *,
    experiment: str,
    method: str,
    seed: int,
    n: int,
    design: Design,
    states: Dict[int, np.ndarray],
    beta_ls: Optional[np.ndarray],
    mu_direction: str,
    eta: float,
    beta_svm: Optional[np.ndarray] = None,
) -> List[Dict]:
    rows: List[Dict] = []
    for step, beta in sorted(states.items()):
        margins = design.K @ beta
        row = base_row(
            experiment=experiment,
            method=method,
            seed=seed,
            n=n,
            spectrum=design.spectrum,
            mu=design.mu,
            mu_direction=mu_direction,
            step=step,
        )
        row["alignment_to_wstar"] = alignment_beta_wstar(beta, design)
        if beta_ls is not None:
            row["alignment_to_ls"] = alignment_beta_beta(beta, beta_ls, design.H)
            row["A_ls_wstar"] = alignment_beta_wstar(beta_ls, design)
        if beta_svm is not None:
            row["alignment_to_svm"] = alignment_beta_beta(beta, beta_svm, design.H)
        row["loss_value"] = (
            logistic_loss_from_margins(margins)
            if method == "logistic"
            else square_loss_from_margins(margins)
        )
        row["min_margin"] = float(np.min(margins))
        row["w_norm_l2"] = beta_norm_K(beta, design.K)
        row["w_norm_sigma"] = beta_norm_H(beta, design.H)
        row["eta"] = eta
        row["gram_condition_number"] = design.condition_number
        rows.append(row)
    return rows


def svm_result_row(
    *,
    experiment: str,
    seed: int,
    n: int,
    design: Design,
    beta_ls: np.ndarray,
    svm: Dict[str, object],
    mu_direction: str,
    step: float | int | str = math.nan,
) -> Dict:
    beta_svm = np.asarray(svm["beta"], dtype=np.float64)
    row = base_row(
        experiment=experiment,
        method="svm",
        seed=seed,
        n=n,
        spectrum=design.spectrum,
        mu=design.mu,
        mu_direction=mu_direction,
        step=step,
    )
    a_svm_wstar = alignment_beta_wstar(beta_svm, design)
    a_ls_wstar = alignment_beta_wstar(beta_ls, design)
    a_svm_ls = alignment_beta_beta(beta_svm, beta_ls, design.H)
    row["alignment_to_wstar"] = a_svm_wstar
    row["alignment_to_ls"] = a_svm_ls
    row["A_svm_wstar"] = a_svm_wstar
    row["A_ls_wstar"] = a_ls_wstar
    row["A_svm_ls"] = a_svm_ls
    row["min_margin"] = float(np.min(design.K @ beta_svm))
    row["gram_condition_number"] = design.condition_number
    gamma = beta_ls
    row["r_min"] = float(np.min(gamma))
    row["prop41_condition_empirical"] = bool(np.all(gamma > 0.0))
    for key in [
        "svm_primal_violation",
        "svm_dual_feasibility_min_alpha",
        "svm_kkt_complementarity",
        "svm_primal_dual_gap",
        "support_vector_fraction",
        "solver_status",
        "solver_nit",
        "solver_success",
        "active_set_polished",
        "active_set_polish_iterations",
    ]:
        row[key] = svm.get(key, np.nan)
    return row


def _category_positions(values: Sequence) -> Tuple[Dict[object, int], List[str]]:
    unique = list(values)
    positions = {v: i for i, v in enumerate(unique)}
    labels = [str(v) for v in unique]
    return positions, labels


def plot_grouped_curve(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str,
    output_base: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    order: Optional[Sequence] = None,
    group_order: Optional[Sequence] = None,
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> None:
    ensure_dir(str(output_base) + ".png")
    data = df.copy()
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[y_col, x_col, group_col])
    if order is None:
        order = sorted(data[x_col].unique(), key=lambda x: float(x) if isinstance(x, (int, float, np.number)) else str(x))
    if group_order is None:
        group_order = list(data[group_col].drop_duplicates())
    pos, labels = _category_positions(order)
    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]
    cmap = plt.get_cmap("tab10")
    for gi, group in enumerate(group_order):
        sub = data[data[group_col] == group]
        if sub.empty:
            continue
        xs, means, stds = [], [], []
        for x in order:
            vals = sub[sub[x_col] == x][y_col].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            xpos = pos[x]
            jitter = (gi - (len(group_order) - 1) / 2.0) * 0.035
            ax.scatter(
                np.full(vals.size, xpos + jitter),
                vals,
                s=14,
                color=cmap(gi % 10),
                alpha=0.28,
                linewidths=0,
            )
            xs.append(xpos)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))
        if xs:
            xs_arr = np.asarray(xs, dtype=float)
            means_arr = np.asarray(means, dtype=float)
            stds_arr = np.asarray(stds, dtype=float)
            ax.plot(
                xs_arr,
                means_arr,
                marker=markers[gi % len(markers)],
                linestyle=linestyles[gi % len(linestyles)],
                color=cmap(gi % 10),
                label=str(group),
                linewidth=1.8,
                markersize=5,
            )
            ax.fill_between(xs_arr, means_arr - stds_arr, means_arr + stds_arr, color=cmap(gi % 10), alpha=0.14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(output_base) + ".png", dpi=350)
    fig.savefig(str(output_base) + ".pdf")
    plt.close(fig)


def plot_multi_panel(
    panels: Sequence[Tuple[pd.DataFrame, str, str, str, Sequence, Sequence]],
    *,
    output_base: str | Path,
    title: str,
    ylabel: str,
    figsize: Tuple[float, float] = (12.0, 7.0),
) -> None:
    ensure_dir(str(output_base) + ".png")
    n_panels = len(panels)
    cols = min(2, n_panels)
    rows = int(math.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for pi, (df, panel_title, x_col, group_col, order, group_order) in enumerate(panels):
        ax = axes[pi // cols][pi % cols]
        data = df.copy()
        data["alignment_to_wstar"] = pd.to_numeric(data["alignment_to_wstar"], errors="coerce")
        pos, labels = _category_positions(order)
        for gi, group in enumerate(group_order):
            sub = data[data[group_col] == group]
            xs, means, stds = [], [], []
            for x in order:
                vals = sub[sub[x_col] == x]["alignment_to_wstar"].dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                xpos = pos[x]
                ax.scatter(np.full(vals.size, xpos), vals, s=12, color=cmap(gi % 10), alpha=0.24, linewidths=0)
                xs.append(xpos)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=0)))
            if xs:
                xs_arr = np.asarray(xs, dtype=float)
                means_arr = np.asarray(means, dtype=float)
                stds_arr = np.asarray(stds, dtype=float)
                ax.plot(xs_arr, means_arr, marker=markers[gi % len(markers)], color=cmap(gi % 10), label=str(group), linewidth=1.6)
                ax.fill_between(xs_arr, means_arr - stds_arr, means_arr + stds_arr, color=cmap(gi % 10), alpha=0.12)
        ax.set_title(panel_title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for pi in range(n_panels, rows * cols):
        axes[pi // cols][pi % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(str(output_base) + ".png", dpi=350)
    fig.savefig(str(output_base) + ".pdf")
    plt.close(fig)


def remove_unexpected_outputs(allowed: Iterable[str]) -> None:
    allowed_set = {os.path.normpath(p) for p in allowed}
    root = Path("outputs")
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and os.path.normpath(str(path)) not in allowed_set:
            path.unlink()
