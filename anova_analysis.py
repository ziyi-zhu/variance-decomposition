#!/usr/bin/env python3
"""
ANOVA judge-effect test and scenario-vs-generation tradeoff analysis.
Produces:
  1. F-test p-values showing judge bias is statistically significant
  2. Scenario-vs-generation tradeoff plot under cycling
"""

import argparse
import json
from pathlib import Path

from run_experiment import JUDGES as _JUDGES_DICT, MODELS as _MODELS_DICT

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

MODELS = list(_MODELS_DICT.keys())
JUDGES = list(_JUDGES_DICT.keys())

MODEL_LABELS = {
    "qwen-2.5-7b-instruct": "Qwen 2.5 7B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "gpt-5.2": "GPT-5.2",
    "gemini-3-flash": "Gemini 3 Flash",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
}
for m in set(MODELS + JUDGES):
    if m not in MODEL_LABELS:
        MODEL_LABELS[m] = m

K_TOT = len(JUDGES)
PLOT_DIR = Path("plots")
DATA_DIR = Path("data")
ALL_BENCHMARKS = ["mt_bench", "mind_eval", "theagentcompany"]


def _load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _score_from_judgment_obj(obj, benchmark: str):
    """Extract single score from judgment JSON. mt_bench: turn1/turn2; mind_eval/theagentcompany: top-level score (fallback: turn1/turn2)."""
    if benchmark in ("mind_eval", "theagentcompany"):
        s = obj.get("score")
        if s is not None:
            return float(s)
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
        return (s1 + s2) / 2.0
    if s1 is not None:
        return s1
    if s2 is not None:
        return s2
    return None


def load_tensor(model, judgements_root: Path, benchmark: str, m_target=None):
    """Load (n, m, K) tensor from cache: {judgements_root}/{model}/{qid}/{index}/{judge}.json.

    If m_target is None, auto-detect the largest m for which at least one
    scenario is complete (same as analyze_results.build_tensor).
    """
    data = {}
    if not judgements_root.exists():
        return np.zeros((0, 0, K_TOT))

    model_dir = judgements_root / model
    if not model_dir.is_dir():
        return np.zeros((0, 0, K_TOT))

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
                    avg = _score_from_judgment_obj(obj, benchmark)
                    if avg is None:
                        continue
                    data.setdefault(question_id, {}).setdefault(index, {})[
                        judge_name
                    ] = avg

    # Auto-detect m_target like analyze_results.build_tensor
    if m_target is None:
        m_candidates = set()
        for qid, gens in data.items():
            for g in range(max(gens.keys()) + 1 if gens else 0):
                if g in gens and all(j in gens[g] for j in JUDGES):
                    m_candidates.add(g + 1)
                else:
                    break
        m_target = max(m_candidates) if m_candidates else 0
    if m_target == 0:
        return np.zeros((0, 0, K_TOT))

    complete = [
        qid
        for qid in sorted(data)
        if all(
            g in data[qid] and all(j in data[qid][g] for j in JUDGES)
            for g in range(m_target)
        )
    ]
    X = np.zeros((len(complete), m_target, K_TOT))
    for i, qid in enumerate(complete):
        for j in range(m_target):
            for l, judge in enumerate(JUDGES):
                X[i, j, l] = data[qid][j][judge]
    return X


def anova_judge_test(X):
    """
    Two-way crossed ANOVA: (scenario*generation) x judge.
    F-test for judge main effect.
    """
    n, m, K = X.shape
    Y = X.reshape(n * m, K)
    N = n * m

    grand = Y.mean()
    judge_m = Y.mean(axis=0)
    subj_m = Y.mean(axis=1)

    SS_J = N * np.sum((judge_m - grand) ** 2)
    SS_S = K * np.sum((subj_m - grand) ** 2)
    SS_T = np.sum((Y - grand) ** 2)
    SS_R = SS_T - SS_J - SS_S

    df_J = K - 1
    df_R = (N - 1) * (K - 1)

    MS_J = SS_J / df_J
    MS_R = SS_R / df_R

    F = MS_J / MS_R
    p = stats.f.sf(F, df_J, df_R)

    return F, p, df_J, df_R


def estimate_components(X):
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
        "sig_ad": sig_ad,
        "sig_beta": sig_beta,
        "sig_eps": sig_eps,
        "sig_gamma": sig_gamma,
    }


def scenario_gen_tradeoff_plot(all_tensors, n_rep=5000, prefix=""):
    """
    For fixed total budget B_gen = n*m, show Var(X-bar) vs budget-per-scenario
    under cycling.  Both scenarios and generations are bootstrapped (with
    replacement), matching the strategy-comparison methodology.

    The bootstrap assigns judge = position % K (not generation_index % K),
    so the variance components must come from the full tensor, not the
    cycling pool.

    Exact prediction (for m divisible by K):
      V(m) = (C_within + m * C_between) / B_gen
      C_between = Var_i( mean over all judges and gens for scenario i )
      C_within  = mean over (scenario, judge) of Var_gen(X[i, :, k])
    """
    models = [m for m in MODELS if m in all_tensors]
    K = K_TOT
    B_gen = 400

    m_values = [mv for mv in [1, 2, 4, 5, 8, 10] if B_gen % mv == 0]

    fig, axes = plt.subplots(1, max(len(models), 2), figsize=(FULL_W, 2.5), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        X = all_tensors[model]
        n_full, m_full, K_full = X.shape

        mu_ik = X.mean(axis=1)                    # (n, K) mean over generations
        mu_i = mu_ik.mean(axis=1)                  # (n,)  mean over gens & judges
        C_between = float(np.var(mu_i))
        C_within = float(np.mean(np.var(X, axis=1)))  # mean gen-variance per (scenario, judge)

        rng = np.random.default_rng(42)
        emp_vars = []
        for mv in m_values:
            n_sub = B_gen // mv
            si = rng.choice(n_full, size=(n_rep, n_sub), replace=True)
            gi = rng.choice(m_full, size=(n_rep, n_sub, mv), replace=True)
            ji = np.arange(mv) % K_full
            vals = X[si[:, :, None], gi, ji[None, None, :]]
            scores = vals.mean(axis=(1, 2))
            emp_vars.append(float(np.var(scores)))

        ax.semilogy(m_values, emp_vars, "o", ms=3, color="C2", label="Empirical")

        m_dense = np.linspace(min(m_values), max(m_values), 200)
        pred_var = (C_within + m_dense * C_between) / B_gen
        ax.semilogy(m_dense, pred_var, "--", lw=1.0, color="C2", label="Predicted")

        if idx == 0:
            ax.set_ylabel("Var(benchmark score)")
        if idx == len(models) // 2:
            ax.set_xlabel(f"Generations per scenario (Total budget = {B_gen})")
        ax.set_title(MODEL_LABELS[model])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    for ax in axes[len(models) :]:
        ax.set_visible(False)
    fname = f"{prefix}scenario_generation_tradeoff.pdf"
    outpath = PLOT_DIR / fname
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def _detect_benchmarks():
    """Return list of benchmarks that have a judgements directory with data."""
    found = []
    for bm in ALL_BENCHMARKS:
        jdir = DATA_DIR / bm / "judgements"
        if jdir.is_dir() and any(jdir.iterdir()):
            found.append(bm)
    return found


def run_benchmark(benchmark: str, n_rep=5000):
    """Run ANOVA F-test and scenario-vs-generation tradeoff for a single benchmark."""
    judgements_root = DATA_DIR / benchmark / "judgements"
    prefix = f"{benchmark}_"

    print(f"\n{'#' * 70}")
    print(f"# Benchmark: {benchmark}")
    print(f"{'#' * 70}")
    print(f"Loading data from {judgements_root}...")

    all_tensors = {}
    all_comps = {}

    for model in MODELS:
        X = load_tensor(model, judgements_root, benchmark)
        all_tensors[model] = X
        n, m, K = X.shape
        if X.shape[0] == 0:
            print(f"  {MODEL_LABELS[model]}: SKIPPED — no data")
            continue
        F, p, df1, df2 = anova_judge_test(X)
        comp = estimate_components(X)
        all_comps[model] = comp

        print(f"\n{MODEL_LABELS[model]} (n={n}, m={m}, K={K}):")
        print(f"  F({df1}, {df2}) = {F:.1f},  p = {p:.2e}")

    if not all_comps:
        print("\nNo models have enough data for this benchmark.")
        return

    print("\n")
    print("=" * 70)
    print(f"SCENARIO vs. GENERATION TRADEOFF [{benchmark}] (under cycling, B_gen=400)")
    print("=" * 70)
    print("\nV(m) = (C_within + m * C_between) / B_gen")
    print("Linear in m => minimized at m=1 (maximize scenarios)\n")

    B_gen = 400
    for model in all_comps:
        X = all_tensors[model]
        n_full, m_full, K_full = X.shape
        mu_ik = X.mean(axis=1)
        mu_i = mu_ik.mean(axis=1)
        C_between = float(np.var(mu_i))
        C_within = float(np.mean(np.var(X, axis=1)))

        var_m1 = (C_within + 1 * C_between) / B_gen
        var_m10 = (C_within + 10 * C_between) / B_gen
        print(f"{MODEL_LABELS[model]}:")
        print(f"  C_between={C_between:.4f}, C_within={C_within:.4f}")
        print(f"  Var(m=1)={var_m1:.6f}, Var(m=10)={var_m10:.6f}")
        print(f"  Ratio Var(m=10)/Var(m=1) = {var_m10 / var_m1:.2f}x")
        print()

    print("Generating scenario-vs-generation tradeoff plot...")
    scenario_gen_tradeoff_plot(all_tensors, n_rep=n_rep, prefix=prefix)
    print(f"Done with {benchmark}.")


def main():
    p = argparse.ArgumentParser(
        description="ANOVA judge-effect test and scenario-vs-generation tradeoff"
    )
    p.add_argument(
        "--benchmark",
        nargs="*",
        default=None,
        help="Benchmark(s) to analyze (default: all with data). "
        "Choices: mt_bench, mind_eval, theagentcompany",
    )
    p.add_argument(
        "--n-rep",
        type=int,
        default=5000,
        help="Number of bootstrap reps for tradeoff plot (default: 5000)",
    )
    args = p.parse_args()

    PLOT_DIR.mkdir(exist_ok=True)

    if args.benchmark is not None:
        benchmarks = args.benchmark
    else:
        benchmarks = _detect_benchmarks()
        if not benchmarks:
            print("No benchmark data found under data/*/judgements/")
            return
        print(f"Auto-detected benchmarks with data: {benchmarks}")

    print("=" * 70)
    print("ANOVA F-TEST FOR JUDGE EFFECTS")
    print("=" * 70)

    for bm in benchmarks:
        run_benchmark(bm, n_rep=args.n_rep)

    print("\nDone.")


if __name__ == "__main__":
    main()
