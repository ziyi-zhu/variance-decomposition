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

from run_experiment import JUDGES as _JUDGES_DICT, MODELS as _MODELS_DICT

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

# ── Models and judges: use current config from run_experiment ─────────────────
MODELS = list(_MODELS_DICT.keys())
JUDGES = list(_JUDGES_DICT.keys())

MODEL_LABELS = {
    "qwen-2.5-7b-instruct": "Qwen 2.5 7B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "gpt-5.2": "GPT-5.2",
}
JUDGE_LABELS = {
    "qwen-2.5-7b-instruct": "Qwen 2.5 7B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "gpt-5.2": "GPT-5.2",
    "gemini-3-flash": "Gemini 3 Flash",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
}
# Fallback: use key as label for any model/judge not in LABELS
for m in MODELS:
    if m not in MODEL_LABELS:
        MODEL_LABELS[m] = m
for j in JUDGES:
    if j not in JUDGE_LABELS:
        JUDGE_LABELS[j] = j

K_TOT = len(JUDGES)

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
JUDGEMENTS_ROOT = DATA_DIR / "mt_bench" / "judgements"

# ── Data Loading ─────────────────────────────────────────────────────────────


def _load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_data_from_cache():
    """Load from new cache: data/mt_bench/judgements/{model}/{qid}/{index}/{judge}.json"""
    data = {}
    if not JUDGEMENTS_ROOT.exists():
        return data
    for model_dir in JUDGEMENTS_ROOT.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for qid_dir in model_dir.iterdir():
            if not qid_dir.is_dir():
                continue
            try:
                question_id = int(qid_dir.name)
            except ValueError:
                continue
            for index_dir in qid_dir.iterdir():
                if not index_dir.is_dir():
                    continue
                try:
                    index = int(index_dir.name)
                except ValueError:
                    continue
                for path in index_dir.iterdir():
                    if path.is_file() and path.suffix == ".json":
                        judge_name = path.stem
                        obj = _load_json(path)
                        if obj is None:
                            continue
                        t1 = (
                            obj.get("turn1")
                            if isinstance(obj.get("turn1"), dict)
                            else None
                        )
                        t2 = (
                            obj.get("turn2")
                            if isinstance(obj.get("turn2"), dict)
                            else None
                        )
                        s1 = t1.get("score") if t1 else None
                        s2 = t2.get("score") if t2 else None
                        if s1 is not None and s2 is not None:
                            avg = (s1 + s2) / 2.0
                        elif s1 is not None:
                            avg = s1
                        elif s2 is not None:
                            avg = s2
                        else:
                            continue
                        data.setdefault(model_name, {}).setdefault(
                            question_id, {}
                        ).setdefault(index, {})[judge_name] = avg
    return data


def load_data():
    """Load from new cache (judgements/) or fallback to all_results.json."""
    data = load_data_from_cache()
    if data:
        return data
    all_path = DATA_DIR / "all_results.json"
    if all_path.exists():
        with open(all_path) as f:
            records = json.load(f)
        for r in records:
            if r["score"] is None:
                continue
            data.setdefault(r["model"], {}).setdefault(r["question_id"], {}).setdefault(
                r["gen_idx"], {}
            )[r["judge"]] = r["score"]
        return data
    raise FileNotFoundError(
        f"No data found. Expected cache under {JUDGEMENTS_ROOT} or {all_path}"
    )


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
        ("\u03bc\u03b8 (mean score)", "mu"),
        ("\u03c3\u00b2_D (dataset)", "sig_ad"),
        ("\u03c3\u00b2_\u03b8 (model)", "sig_beta"),
        ("\u03c3\u00b2\u03b5 (residual)", "sig_eps"),
        ("\u03c3\u00b2\u03b3 (judge bias)", "sig_gamma"),
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
    lines.append("Judge biases (\u03b3\u0302):")
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
    print("\n  Saved results_summary.txt")


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_judge_biases(all_comp):
    """Heatmap of judge biases for each model."""
    models = [m for m in MODELS if m in all_comp]
    mat = np.zeros((len(JUDGES), len(models)))
    for jm, model in enumerate(models):
        for jj, judge in enumerate(JUDGES):
            mat[jj, jm] = all_comp[model]["gamma"][judge]

    # Standard single-column size (3.5" width); larger figure => text appears smaller
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
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


def compute_prediction_constants(X):
    """Compute exact V(B) = C/(nB) constants directly from the data tensor.

    For bootstrap with replacement, each draw is i.i.d. from the empirical
    pool, so Var(mean of m draws) = pool_variance / m.  The three constants
    are the pool variances relevant to each strategy:
      all_judges:  pool = generation-level means (averaged over K judges)
      random_1:    pool = all individual cells (scenario x gen x judge)
      cycle:       pool = within-judge columns (one per judge per scenario)
    """
    n, m_max, K = X.shape
    gen_means = X.mean(axis=2)  # (n, m_max)
    C_all = K * float(gen_means.var(axis=1).mean())
    C_random = float(X.reshape(n, -1).var(axis=1).mean())
    C_cycle = float(X.var(axis=1).mean())
    return C_all, C_random, C_cycle


def plot_strategy_comparison(all_tensors, n_rep=5000):
    """Empirical variance of benchmark score vs budget for three strategies,
    overlaid with exact theoretical predictions (dashed)."""
    models = [m for m in MODELS if m in all_tensors]
    K = K_TOT
    max_budget = all_tensors[models[0]].shape[1] * K

    budgets_k = list(range(K, max_budget + 1, K))

    fig, axes = plt.subplots(1, max(len(models), 2), figsize=(FULL_W, 2.5), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        X = all_tensors[model]
        n = X.shape[0]
        C_all, C_random, C_cycle = compute_prediction_constants(X)

        var_allj, var_rand, var_cycle = [], [], []
        for B in budgets_k:
            var_allj.append(
                np.var(
                    subsample_benchmark(
                        X, "all_judges", B, n_rep, np.random.default_rng(42)
                    )
                )
            )
            var_rand.append(
                np.var(
                    subsample_benchmark(
                        X, "random_1", B, n_rep, np.random.default_rng(42)
                    )
                )
            )
            var_cycle.append(
                np.var(
                    subsample_benchmark(X, "cycle", B, n_rep, np.random.default_rng(42))
                )
            )

        ax.semilogy(budgets_k, var_allj, "o", ms=3, color="C0", label=f"All {K} judges")
        ax.semilogy(budgets_k, var_rand, "^", ms=3, color="C1", label="Random 1 judge")
        ax.semilogy(
            budgets_k, var_cycle, "s", ms=3, color="C2", label=f"Cycle {K} judges"
        )

        B_dense = np.linspace(K, max_budget, 200)
        ax.semilogy(B_dense, C_all / (n * B_dense), "--", lw=1.0, color="C0")
        ax.semilogy(B_dense, C_random / (n * B_dense), "--", lw=1.0, color="C1")
        ax.semilogy(B_dense, C_cycle / (n * B_dense), "--", lw=1.0, color="C2")

        if idx == 0:
            ax.set_ylabel("Var(benchmark score)")
            ax.plot([], [], "--", color="gray", lw=0.8, label="Theory")
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
        ncol=4,
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
    p.add_argument("--quick", action="store_true", help="Fewer reps for fast check")
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
            print("  SKIPPED — no data")
            continue
        X, qids = build_tensor(md)
        if X.shape[0] == 0:
            print("  SKIPPED — no complete scenarios")
            continue

        print(f"  Tensor: {X.shape} (n={X.shape[0]}, m={X.shape[1]}, K={X.shape[2]})")

        comp = estimate_components(X)
        all_comp[model] = comp
        all_ci[model] = bootstrap_ci(X, B=bootstrap_B)
        all_tensors[model] = X

        print(
            f"  \u03bc\u03b8={comp['mu']:.4f}  \u03c3\u00b2_D={comp['sig_ad']:.4f}  "
            f"\u03c3\u00b2_\u03b8={comp['sig_beta']:.4f}  \u03c3\u00b2\u03b5={comp['sig_eps']:.4f}  "
            f"\u03c3\u00b2\u03b3={comp['sig_gamma']:.4f}"
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
