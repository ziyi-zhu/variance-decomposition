#!/usr/bin/env python3
"""
ANOVA judge-effect test and scenario-vs-generation tradeoff analysis.
Produces:
  1. F-test p-values showing judge bias is statistically significant
  2. Scenario-vs-generation tradeoff plot under cycling
"""

import json
from pathlib import Path

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
PLOT_DIR = Path("plots")


def load_tensor(model, m_target=10):
    with open("data/judgments.json") as f:
        jdg_cache = json.load(f)

    turn_scores = {}
    for key, val in jdg_cache.items():
        parts = key.split("|")
        score = val.get("score") if isinstance(val, dict) else val
        if score is None:
            continue
        if len(parts) == 5:
            m, qid_str, gen_str, judge, turn_key = parts
            if m != model:
                continue
            base = (int(qid_str), int(gen_str), judge)
            turn_scores.setdefault(base, {})[turn_key] = score

    data = {}
    for (qid, gen_idx, judge), turns in turn_scores.items():
        s1, s2 = turns.get("t1"), turns.get("t2")
        if s1 is not None and s2 is not None:
            avg = (s1 + s2) / 2.0
        else:
            continue
        data.setdefault(qid, {}).setdefault(gen_idx, {})[judge] = avg

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


def scenario_gen_tradeoff_plot(all_tensors, n_rep=5000):
    """
    For fixed total budget B_gen = n*m, show Var(X-bar) vs budget-per-scenario
    under cycling.  Both scenarios and generations are bootstrapped (with
    replacement), matching the strategy-comparison methodology.

    Exact prediction: V(m) = (C_within + m * C_between) / B_gen
      C_between = pop. variance of scenario-level cycling-pool means
      C_within  = avg within-scenario cycling-pool variance
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

        cycling_pool = np.zeros((n_full, m_full))
        for g in range(m_full):
            cycling_pool[:, g] = X[:, g, g % K_full]

        scenario_means = cycling_pool.mean(axis=1)
        C_between = float(np.var(scenario_means))
        C_within = float(np.mean(np.var(cycling_pool, axis=1)))

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
    outpath = PLOT_DIR / "scenario_generation_tradeoff.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


if __name__ == "__main__":
    PLOT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("ANOVA F-TEST FOR JUDGE EFFECTS")
    print("=" * 70)

    all_tensors = {}
    all_comps = {}

    for model in MODELS:
        X = load_tensor(model)
        all_tensors[model] = X
        n, m, K = X.shape
        F, p, df1, df2 = anova_judge_test(X)
        comp = estimate_components(X)
        all_comps[model] = comp

        print(f"\n{MODEL_LABELS[model]} (n={n}, m={m}, K={K}):")
        print(f"  F({df1}, {df2}) = {F:.1f},  p = {p:.2e}")

    print("\n")
    print("=" * 70)
    print("SCENARIO vs. GENERATION TRADEOFF (under cycling, B_gen=400)")
    print("=" * 70)
    print("\nV(m) = (C_within + m * C_between) / B_gen")
    print("Linear in m => minimized at m=1 (maximize scenarios)\n")

    B_gen = 400
    for model in MODELS:
        X = all_tensors[model]
        n_full, m_full, K = X.shape
        cycling_pool = np.zeros((n_full, m_full))
        for g in range(m_full):
            cycling_pool[:, g] = X[:, g, g % K]
        scenario_means = cycling_pool.mean(axis=1)
        C_between = float(np.var(scenario_means))
        C_within = float(np.mean(np.var(cycling_pool, axis=1)))

        var_m1 = (C_within + 1 * C_between) / B_gen
        var_m10 = (C_within + 10 * C_between) / B_gen
        print(f"{MODEL_LABELS[model]}:")
        print(f"  C_between={C_between:.4f}, C_within={C_within:.4f}")
        print(f"  Var(m=1)={var_m1:.6f}, Var(m=10)={var_m10:.6f}")
        print(f"  Ratio Var(m=10)/Var(m=1) = {var_m10 / var_m1:.2f}x")
        print()

    print("Generating scenario-vs-generation tradeoff plot...")
    scenario_gen_tradeoff_plot(all_tensors, n_rep=5000)
    print("\nDone.")
