#!/usr/bin/env python3
"""
Variance Decomposition Analysis
================================
ANOVA-based variance component estimation with subsampling-based
strategy comparison.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ACL Plot Style ───────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)

COL_W = 3.25
FULL_W = 6.75

MODELS = ["claude-haiku-4.5", "llama-3.3-70b-instruct", "gpt-5.2-chat"]
MODEL_LABELS = {
    "claude-haiku-4.5": "Haiku 4.5",
    "llama-3.3-70b-instruct": "Llama 3.3-70B",
    "gpt-5.2-chat": "GPT-5.2",
}
JUDGES = [
    "claude-sonnet-4.5",
    "gpt-5.2",
    "gemini-3-flash",
    "kimi-k2",
    "llama-3.3-70b-instruct",
]
JUDGE_LABELS = {
    "claude-sonnet-4.5": "Sonnet 4.5",
    "gpt-5.2": "GPT-5.2",
    "gemini-3-flash": "Gemini 3 Flash",
    "kimi-k2": "Kimi K2",
    "llama-3.3-70b-instruct": "Llama 3.3-70B",
}
K_TOT = len(JUDGES)

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")

# ── Data Loading ─────────────────────────────────────────────────────────────


def load_data():
    """Load from checkpoint files."""
    jdg_path = DATA_DIR / "judgments.json"
    all_path = DATA_DIR / "all_results.json"

    if jdg_path.exists():
        with open(jdg_path) as f:
            jdg_cache = json.load(f)

        turn_scores = {}
        legacy_data = {}

        for key, val in jdg_cache.items():
            parts = key.split("|")
            score = val.get("score") if isinstance(val, dict) else val
            if score is None:
                continue

            if len(parts) == 5:
                model, qid_str, gen_str, judge, turn_key = parts
                base = (model, int(qid_str), int(gen_str), judge)
                turn_scores.setdefault(base, {})[turn_key] = score
            elif len(parts) == 4:
                model, qid_str, gen_str, judge = parts
                legacy_data.setdefault(model, {}).setdefault(
                    int(qid_str), {}
                ).setdefault(int(gen_str), {})[judge] = score

        if turn_scores:
            data = {}
            for (model, qid, gen_idx, judge), turns in turn_scores.items():
                s1, s2 = turns.get("t1"), turns.get("t2")
                if s1 is not None and s2 is not None:
                    avg = (s1 + s2) / 2.0
                elif s1 is not None:
                    avg = s1
                elif s2 is not None:
                    avg = s2
                else:
                    continue
                data.setdefault(model, {}).setdefault(qid, {}).setdefault(gen_idx, {})[
                    judge
                ] = avg
            return data
        return legacy_data

    if all_path.exists():
        with open(all_path) as f:
            records = json.load(f)
        data = {}
        for r in records:
            if r["score"] is None:
                continue
            data.setdefault(r["model"], {}).setdefault(r["question_id"], {}).setdefault(
                r["gen_idx"], {}
            )[r["judge"]] = r["score"]
        return data

    raise FileNotFoundError(f"No data files found. Expected {jdg_path} or {all_path}")


def build_tensor(model_data, m_target=10):
    """Build balanced (n, m, K) tensor; drop incomplete scenarios."""
    complete = []
    for qid in sorted(model_data.keys()):
        gens = model_data[qid]
        if all(
            g in gens and all(j in gens[g] for j in JUDGES) for g in range(m_target)
        ):
            complete.append(qid)
    n = len(complete)
    scores = np.zeros((n, m_target, K_TOT))
    for i, qid in enumerate(complete):
        for j in range(m_target):
            for l, judge in enumerate(JUDGES):
                scores[i, j, l] = model_data[qid][j][judge]
    return scores, complete


# ── ANOVA Estimation ─────────────────────────────────────────────────────────


def estimate_components(X):
    """Method-of-moments ANOVA on (n, m, K) array."""
    n, m, K = X.shape
    Xbar = X.mean()
    Xbar_ij = X.mean(axis=2)
    Xbar_i = X.mean(axis=(1, 2))
    Xbar_l = X.mean(axis=(0, 1))

    resid = X - Xbar_ij[:, :, None] - Xbar_l[None, None, :] + Xbar
    MS_W = np.sum(resid**2) / ((n * m - 1) * (K - 1))
    MS_G = K * np.sum((Xbar_ij - Xbar_i[:, None]) ** 2) / (n * (m - 1))
    MS_S = m * K * np.sum((Xbar_i - Xbar) ** 2) / max(n - 1, 1)

    sig_eps = MS_W
    sig_beta = max(0.0, (MS_G - MS_W) / K)
    sig_ad = max(0.0, (MS_S - MS_G) / (m * K))

    gamma = Xbar_l - Xbar
    sig_gamma = max(0.0, np.mean(gamma**2) - sig_eps / (n * m) * (K - 1) / K)

    return {
        "mu": float(Xbar),
        "sig_eps": float(sig_eps),
        "sig_beta": float(sig_beta),
        "sig_ad": float(sig_ad),
        "sig_gamma": float(sig_gamma),
        "gamma": {JUDGES[l]: float(gamma[l]) for l in range(K)},
        "n": n,
        "m": m,
        "K": K,
    }


def bootstrap_ci(X, B=2000, alpha=0.05):
    rng = np.random.default_rng(42)
    n = X.shape[0]
    keys = ["mu", "sig_eps", "sig_beta", "sig_ad", "sig_gamma"]
    samples = {k: [] for k in keys}
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        est = estimate_components(X[idx])
        for k in keys:
            samples[k].append(est[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: (float(np.percentile(samples[k], lo)), float(np.percentile(samples[k], hi)))
        for k in keys
    }


# ── Decomposition Table ─────────────────────────────────────────────────────


def print_decomposition_table(all_comp, all_ci):
    """Print and save variance decomposition as a formatted table."""
    models = [m for m in MODELS if m in all_comp]

    lines = []
    lines.append("=" * 92)
    lines.append("VARIANCE DECOMPOSITION")
    lines.append("=" * 92)

    header = f"{'Component':<28s}"
    for m in models:
        header += f" | {MODEL_LABELS[m]:>18s}"
    lines.append(header)
    lines.append("-" * 92)

    rows = [
        ("μ (mean score)", "mu"),
        ("σ²(α+δ) scenario", "sig_ad"),
        ("σ²β generation", "sig_beta"),
        ("σ²ε judge noise", "sig_eps"),
        ("σ²γ judge bias", "sig_gamma"),
    ]
    for label, key in rows:
        line = f"{label:<28s}"
        for m in models:
            val = all_comp[m][key]
            ci_lo, ci_hi = all_ci[m][key]
            line += f" | {val:7.4f} [{ci_lo:.4f},{ci_hi:.4f}]"
        lines.append(line)

    lines.append("-" * 92)
    lines.append("")
    lines.append("Judge biases (γ̂):")
    jh = f"{'Judge':<20s}"
    for m in models:
        jh += f" | {MODEL_LABELS[m]:>12s}"
    lines.append(jh)
    lines.append("-" * 62)
    for judge in JUDGES:
        jl = f"{JUDGE_LABELS[judge]:<20s}"
        for m in models:
            jl += f" | {all_comp[m]['gamma'][judge]:>+12.4f}"
        lines.append(jl)

    txt = "\n".join(lines)
    print("\n" + txt)
    Path("results_summary.txt").write_text(txt)
    print(f"\n  Saved results_summary.txt")


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_judge_biases(all_comp):
    """Heatmap of judge biases for each model."""
    models = [m for m in MODELS if m in all_comp]
    mat = np.zeros((len(JUDGES), len(models)))
    for jm, model in enumerate(models):
        for jj, judge in enumerate(JUDGES):
            mat[jj, jm] = all_comp[model]["gamma"][judge]

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    vmax = max(abs(mat.min()), abs(mat.max())) or 0.1
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0)
    ax.set_yticks(range(len(JUDGES)))
    ax.set_yticklabels([JUDGE_LABELS[j] for j in JUDGES])
    ax.set_xlabel("Evaluated model")
    ax.set_ylabel("Judge model")
    for jj in range(len(JUDGES)):
        for jm in range(len(models)):
            ax.text(
                jm,
                jj,
                f"{mat[jj, jm]:+.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(mat[jj, jm]) > vmax * 0.6 else "black",
            )
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(r"Judge bias $\hat\gamma_\ell$", fontsize=8)
    fig.savefig(PLOT_DIR / "judge_biases.pdf")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'judge_biases.pdf'}")


def subsample_benchmark(X, strategy, budget, n_rep, rng):
    """Run n_rep subsampling reps, return (n_rep,) array of benchmark scores.

    Budget = total judge evaluations per scenario.
    Generations sampled with replacement; benchmark score = grand mean over
    all scenarios.
    """
    n, m_max, K = X.shape
    si = np.arange(n)[None, :, None]  # (1, n, 1) for broadcasting

    if strategy == "all_judges":
        m = budget // K
        gi = rng.choice(m_max, size=(n_rep, n, m), replace=True)
        sampled = X[si, gi, :]  # (n_rep, n, m, K)
        return sampled.mean(axis=(2, 3)).mean(axis=1)

    if strategy == "random_1":
        m = budget
        gi = rng.choice(m_max, size=(n_rep, n, m), replace=True)
        ji = rng.integers(K, size=(n_rep, n, m))
        sampled = X[si, gi, ji]  # (n_rep, n, m)
        return sampled.mean(axis=2).mean(axis=1)

    if strategy == "cycle":
        m = budget
        gi = rng.choice(m_max, size=(n_rep, n, m), replace=True)
        ji = np.broadcast_to(np.arange(m) % K, (n_rep, n, m))
        sampled = X[si, gi, ji]  # (n_rep, n, m)
        return sampled.mean(axis=2).mean(axis=1)

    raise ValueError(f"Unknown strategy: {strategy}")


def plot_strategy_comparison(all_tensors, n_rep=5000):
    """Empirical variance of benchmark score vs budget for three strategies."""
    models = [m for m in MODELS if m in all_tensors]
    K = K_TOT
    max_budget = all_tensors[models[0]].shape[1] * K  # m_max * K

    budgets_k = list(range(K, max_budget + 1, K))
    budgets_all = list(range(1, max_budget + 1))

    fig, axes = plt.subplots(
        1, max(len(models), 2), figsize=(FULL_W, 2.5), sharey=True
    )
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        X = all_tensors[model]

        var_allj = []
        for B in budgets_k:
            scores = subsample_benchmark(X, "all_judges", B, n_rep, np.random.default_rng(42))
            var_allj.append(np.var(scores))

        var_rand = []
        for B in budgets_all:
            scores = subsample_benchmark(X, "random_1", B, n_rep, np.random.default_rng(42))
            var_rand.append(np.var(scores))

        var_cycle = []
        for B in budgets_k:
            scores = subsample_benchmark(X, "cycle", B, n_rep, np.random.default_rng(42))
            var_cycle.append(np.var(scores))

        ax.semilogy(
            budgets_k, var_allj, "-o", ms=3, lw=1.2, label=f"All {K} judges/gen"
        )
        ax.semilogy(
            budgets_all, var_rand, "-", lw=1.2, alpha=0.8, label="Random 1 judge/gen"
        )
        ax.semilogy(
            budgets_k, var_cycle, "-s", ms=3, lw=1.2, label=f"Cycle {K} judges"
        )

        if idx == 0:
            ax.set_ylabel("Var(benchmark score)")
        if idx == len(models) // 2:
            ax.set_xlabel("Budget per scenario")
        ax.set_title(MODEL_LABELS[model])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    for ax in axes[len(models) :]:
        ax.set_visible(False)
    fig.savefig(PLOT_DIR / "strategy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'strategy_comparison.pdf'}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Variance decomposition analysis")
    p.add_argument(
        "--quick", action="store_true", help="Fewer reps for fast check"
    )
    args = p.parse_args()

    PLOT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    data = load_data()

    all_comp, all_ci, all_tensors = {}, {}, {}
    n_rep = 500 if args.quick else 5000
    bootstrap_B = 100 if args.quick else 2000

    for model in MODELS:
        print(f"\nAnalyzing: {model}")
        md = data.get(model, {})
        if not md:
            print(f"  SKIPPED — no data")
            continue
        X, qids = build_tensor(md)
        if X.shape[0] == 0:
            print(f"  SKIPPED — no complete scenarios")
            continue

        print(
            f"  Tensor: {X.shape} (n={X.shape[0]}, m={X.shape[1]}, K={X.shape[2]})"
        )

        comp = estimate_components(X)
        all_comp[model] = comp
        all_ci[model] = bootstrap_ci(X, B=bootstrap_B)
        all_tensors[model] = X

        print(
            f"  μ={comp['mu']:.4f}  σ²ε={comp['sig_eps']:.4f}  "
            f"σ²β={comp['sig_beta']:.4f}  σ²(α+δ)={comp['sig_ad']:.4f}  "
            f"σ²γ={comp['sig_gamma']:.4f}"
        )

    if not all_comp:
        print("\nNo models have enough data.")
        return

    print_decomposition_table(all_comp, all_ci)
    plot_judge_biases(all_comp)

    print(f"\nRunning strategy comparison ({n_rep} reps per budget point)...")
    plot_strategy_comparison(all_tensors, n_rep=n_rep)


if __name__ == "__main__":
    main()
