#!/usr/bin/env python3
"""
Variance Decomposition Analysis
================================
Implements ANOVA-based variance component estimation, bootstrap CIs,
D-study predictions, subsampling validation, and ACL-style PDF plots.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── ACL Plot Style ───────────────────────────────────────────────────────────

plt.rcParams.update({
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
})

COL_W = 3.25   # ACL single-column width (inches)
FULL_W = 6.75  # ACL full-page width (inches)

MODELS = ["claude-sonnet-4.5", "llama-3.3-70b", "gpt-5.2-chat"]
MODEL_LABELS = {"claude-sonnet-4.5": "Sonnet 4.5", "llama-3.3-70b": "Llama 3.3-70B", "gpt-5.2-chat": "GPT-5.2"}
JUDGES = ["claude-opus-4.6", "gpt-5.2", "gemini-3-pro", "kimi-k2.5", "glm-5"]
JUDGE_LABELS = {"claude-opus-4.6": "Opus 4.6", "gpt-5.2": "GPT-5.2",
                "gemini-3-pro": "Gemini 3", "kimi-k2.5": "Kimi K2.5", "glm-5": "GLM-5"}
K_TOT = len(JUDGES)

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")

COLORS = {
    "scenario": "#2176AE",
    "generation": "#57B8FF",
    "judge_noise": "#B66D0D",
    "judge_bias": "#FBB13C",
}
MODEL_COLORS = ["#2176AE", "#E84855", "#44AF69"]

# ── Data Loading ─────────────────────────────────────────────────────────────

def _load_generation_stats():
    """Count generations per model from the checkpoint file."""
    gen_path = DATA_DIR / "generations.json"
    if not gen_path.exists():
        return {}
    with open(gen_path) as f:
        gen_cache = json.load(f)
    stats = {}
    for key in gen_cache:
        parts = key.split("|")
        if len(parts) != 3:
            continue
        model, qid_str, gen_str = parts
        s = stats.setdefault(model, {"questions": set(), "total": 0})
        s["questions"].add(int(qid_str))
        s["total"] += 1
    return stats


def load_data():
    """Load from checkpoint files so analysis can run mid-experiment.

    Falls back to the final all_results.json if the checkpoint files
    don't exist (e.g. after a completed run that cleaned up).
    """
    jdg_path = DATA_DIR / "judgments.json"
    all_path = DATA_DIR / "all_results.json"

    gen_stats = _load_generation_stats()
    if gen_stats:
        print("  Generation progress:")
        for model in MODELS:
            s = gen_stats.get(model)
            if s:
                print(f"    {model}: {s['total']} gens across "
                      f"{len(s['questions'])} questions")
            else:
                print(f"    {model}: no generations yet")

    if jdg_path.exists():
        with open(jdg_path) as f:
            jdg_cache = json.load(f)
        data = {}
        jdg_counts = {}
        for key, val in jdg_cache.items():
            parts = key.split("|")
            if len(parts) != 4:
                continue
            model, qid_str, gen_str, judge = parts
            score = val.get("score") if isinstance(val, dict) else val
            if score is None:
                continue
            qid = int(qid_str)
            gen_idx = int(gen_str)
            data.setdefault(model, {}).setdefault(qid, {}).setdefault(
                gen_idx, {}
            )[judge] = score
            jdg_counts[model] = jdg_counts.get(model, 0) + 1

        total_scores = sum(jdg_counts.values())
        print(f"  Judgment progress ({total_scores} scores):")
        for model in MODELS:
            n_jdg = jdg_counts.get(model, 0)
            n_gen = gen_stats.get(model, {}).get("total", 0)
            expected = n_gen * K_TOT
            pct = n_jdg / expected * 100 if expected else 0
            print(f"    {model}: {n_jdg}/{expected} judgments "
                  f"({pct:.0f}%)")
        return data

    if all_path.exists():
        with open(all_path) as f:
            records = json.load(f)
        data = {}
        for r in records:
            s = r["score"]
            if s is None:
                continue
            data.setdefault(r["model"], {}).setdefault(
                r["question_id"], {}
            ).setdefault(r["gen_idx"], {})[r["judge"]] = s
        print(f"  Loaded from all_results.json")
        return data

    raise FileNotFoundError(
        f"No data files found. Expected {jdg_path} or {all_path}"
    )


def build_tensor(model_data, m_target=10):
    """Build balanced (n, m, K) tensor; drop incomplete scenarios."""
    complete = []
    total_scenarios = len(model_data)
    for qid in sorted(model_data.keys()):
        gens = model_data[qid]
        ok = True
        for g in range(m_target):
            if g not in gens:
                ok = False; break
            if any(j not in gens[g] for j in JUDGES):
                ok = False; break
        if ok:
            complete.append(qid)
    n = len(complete)
    if n == 0:
        print(f"  WARNING: 0/{total_scenarios} scenarios fully complete "
              f"(need {m_target} gens × {K_TOT} judges each)")
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
    Xbar_ij = X.mean(axis=2)          # (n, m)
    Xbar_i  = X.mean(axis=(1, 2))     # (n,)
    Xbar_l  = X.mean(axis=(0, 1))     # (K,)

    # Residual (cell × judge interaction)
    resid = X - Xbar_ij[:, :, None] - Xbar_l[None, None, :] + Xbar
    MS_W = np.sum(resid**2) / ((n * m - 1) * (K - 1))

    # Between-generation within scenario
    MS_G = K * np.sum((Xbar_ij - Xbar_i[:, None])**2) / (n * (m - 1))

    # Between-scenario
    MS_S = m * K * np.sum((Xbar_i - Xbar)**2) / max(n - 1, 1)

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
        "n": n, "m": m, "K": K,
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
    return {k: (float(np.percentile(samples[k], lo)),
                float(np.percentile(samples[k], hi))) for k in keys}

# ── D-Study / Prediction ────────────────────────────────────────────────────

def predict_var(sig_ad, sig_beta, sig_eps, sig_gamma, n, m, K, K_tot):
    """Full benchmark variance prediction including scenario component."""
    fpc = (K_tot - K) / (K_tot - 1) if K_tot > 1 and K < K_tot else 0.0
    return sig_ad / n + sig_beta / (n * m) + sig_eps / (n * m * K) + sig_gamma / K * fpc


def predict_per_scenario_var(sig_beta, sig_eps, sig_gamma, n, m, K, K_tot):
    """Per-scenario variance excluding the scenario component.

    When subsampling from fixed scenarios, only generation/judge/bias
    components vary across repetitions.  This is the correct prediction
    target for hold-scenario-fixed validation.
    """
    fpc = (K_tot - K) / (K_tot - 1) if K_tot > 1 and K < K_tot else 0.0
    return sig_beta / (n * m) + sig_eps / (n * m * K) + sig_gamma / K * fpc


def subsample_validation(X, comp, n_rep=10000):
    """Compare predicted vs empirical variance over all (K,m) designs.

    Scenarios are held fixed.  Judges sampled w/o replacement (finite
    pool → FPC).  Generations sampled w/ replacement (approximates
    i.i.d. draws from the model distribution).
    """
    n, m_max, K_max = X.shape
    rng = np.random.default_rng(123)
    rows = []
    for K in range(1, K_max + 1):
        for m in range(1, m_max + 1):
            pred = predict_per_scenario_var(
                comp["sig_beta"], comp["sig_eps"], comp["sig_gamma"],
                n, m, K, K_max,
            )
            means = np.empty(n_rep)
            for r in range(n_rep):
                jj = rng.choice(K_max, size=K, replace=False)
                s = 0.0
                for i in range(n):
                    gi = rng.choice(m_max, size=m, replace=True)
                    s += X[i][np.ix_(gi, jj)].mean()
                means[r] = s / n
            actual = float(np.var(means, ddof=0))
            rows.append({"K": K, "m": m, "B": K * m,
                          "predicted": pred, "actual": actual})
    return rows

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_variance_components(all_comp, all_ci):
    """Grouped bar chart of variance components across models."""
    models = [m for m in MODELS if m in all_comp]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    comps = ["sig_ad", "sig_beta", "sig_eps", "sig_gamma"]
    labels = [r"$\hat\sigma^2_{\alpha+\delta}$ (scenario)",
              r"$\hat\sigma^2_\beta$ (generation)",
              r"$\hat\sigma^2_\varepsilon$ (judge noise)",
              r"$\hat\sigma^2_\gamma$ (judge bias)"]
    clist = [COLORS["scenario"], COLORS["generation"],
             COLORS["judge_noise"], COLORS["judge_bias"]]

    x = np.arange(len(models))
    w = 0.18
    offsets = np.arange(len(comps)) - (len(comps) - 1) / 2

    for ic, comp_key in enumerate(comps):
        vals = [all_comp[m][comp_key] for m in models]
        lo = [vals[j] - all_ci[m][comp_key][0] for j, m in enumerate(models)]
        hi = [all_ci[m][comp_key][1] - vals[j] for j, m in enumerate(models)]
        ax.bar(x + offsets[ic] * w, vals, w * 0.9, label=labels[ic],
               color=clist[ic], yerr=[lo, hi], capsize=2, error_kw={"lw": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Variance component estimate")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    fig.savefig(PLOT_DIR / "variance_components.pdf")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'variance_components.pdf'}")


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
    for jj in range(len(JUDGES)):
        for jm in range(len(models)):
            ax.text(jm, jj, f"{mat[jj, jm]:+.3f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(mat[jj, jm]) > vmax * 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Judge bias " + r"$\hat\gamma_\ell$", fontsize=8)
    fig.savefig(PLOT_DIR / "judge_biases.pdf")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'judge_biases.pdf'}")


def plot_predicted_vs_actual(all_val):
    """Scatter: predicted vs actual variance for all (K,m) designs."""
    models = [m for m in MODELS if m in all_val]
    fig, axes = plt.subplots(1, max(len(models), 2), figsize=(FULL_W, 2.3), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for idx, model in enumerate(models):
        ax = axes[idx]
        rows = all_val[model]
        pred = np.array([r["predicted"] for r in rows])
        act  = np.array([r["actual"] for r in rows])
        budgets = np.array([r["B"] for r in rows])

        pos = (pred > 0) & (act > 0)
        sc = ax.scatter(pred[pos], act[pos], c=budgets[pos], cmap="viridis",
                        s=28, edgecolors="k", linewidths=0.3, zorder=3)
        if pos.any():
            lo = min(pred[pos].min(), act[pos].min()) * 0.5
            hi = max(pred[pos].max(), act[pos].max()) * 2.0
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.5)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xscale("log"); ax.set_yscale("log")

        r2 = _r2(pred[pos], act[pos]) if pos.sum() > 2 else float("nan")
        ax.text(0.05, 0.92, f"$R^2$={r2:.3f}", transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("Predicted variance")
        if idx == 0:
            ax.set_ylabel("Actual variance (subsampled)")
        ax.set_title(MODEL_LABELS[model])

    for ax in axes[len(models):]:
        ax.set_visible(False)
    fig.colorbar(sc, ax=axes[:len(models)], label="Budget $B = mK$", shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "predicted_vs_actual.pdf")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'predicted_vs_actual.pdf'}")


def _r2(pred, actual):
    ss_res = np.sum((pred - actual)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def plot_strategy_comparison(all_comp):
    """Per-scenario variance vs budget for all-judges vs max-gens strategies."""
    models = [m for m in MODELS if m in all_comp]
    fig, axes = plt.subplots(1, max(len(models), 2), figsize=(FULL_W, 2.3), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    budgets = np.arange(1, 26)

    for idx, model in enumerate(models):
        ax = axes[idx]
        c = all_comp[model]

        var_allj, var_maxg, var_opt = [], [], []
        for B in budgets:
            # All-judges: K=K_TOT, m=B/K_TOT (fractional ok for prediction)
            m_aj = B / K_TOT
            v_aj = predict_var(c["sig_ad"], c["sig_beta"], c["sig_eps"],
                               c["sig_gamma"], c["n"], m_aj, K_TOT, K_TOT)
            var_allj.append(v_aj)

            # Max-generations: K=1, m=B
            v_mg = predict_var(c["sig_ad"], c["sig_beta"], c["sig_eps"],
                               c["sig_gamma"], c["n"], B, 1, K_TOT)
            var_maxg.append(v_mg)

            # Optimal over all integer K
            best = np.inf
            for K in range(1, min(K_TOT, B) + 1):
                if B % K != 0:
                    continue
                m_k = B // K
                v = predict_var(c["sig_ad"], c["sig_beta"], c["sig_eps"],
                                c["sig_gamma"], c["n"], m_k, K, K_TOT)
                best = min(best, v)
            var_opt.append(best)

        ax.semilogy(budgets, var_allj, "-o", ms=3, lw=1.2, label=f"All judges ($K$={K_TOT})")
        ax.semilogy(budgets, var_maxg, "-s", ms=3, lw=1.2, label="Max gens ($K$=1)")
        ax.semilogy(budgets, var_opt, "-^", ms=3, lw=1.2, label="Optimal $K$", color="green")

        # Mark break-even
        if c["sig_gamma"] > 0 and c["sig_beta"] > 0:
            B_star = (K_TOT - 1) * c["sig_beta"] / c["sig_gamma"]
            if 0 < B_star < 25:
                ax.axvline(B_star, ls=":", color="gray", lw=0.8)
                ylims = ax.get_ylim()
                ax.text(B_star + 0.5, ylims[0] * (ylims[1]/ylims[0])**0.1,
                        f"$B^*$={B_star:.0f}", fontsize=7, color="gray")

        ax.set_xlabel("Per-scenario budget $B$")
        if idx == 0:
            ax.set_ylabel("Total benchmark variance")
        ax.set_title(MODEL_LABELS[model])
        if idx == 2:
            ax.legend(frameon=False, fontsize=7, loc="upper right")

    for ax in axes[len(models):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "strategy_comparison.pdf")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'strategy_comparison.pdf'}")


def plot_decomposition_stacked(all_comp):
    """Stacked variance decomposition at operating point for each model."""
    models = [m for m in MODELS if m in all_comp]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    comps_keys = ["sig_ad", "sig_beta", "sig_eps", "sig_gamma"]
    comp_labels = ["Scenario", "Generation", "Judge noise", "Judge bias"]
    clist = [COLORS["scenario"], COLORS["generation"],
             COLORS["judge_noise"], COLORS["judge_bias"]]

    x = np.arange(len(models))
    bottom = np.zeros(len(models))

    for ic, ck in enumerate(comps_keys):
        vals = []
        for model in models:
            c = all_comp[model]
            n, m, K = c["n"], c["m"], c["K"]
            if ck == "sig_ad":
                v = c["sig_ad"] / n
            elif ck == "sig_beta":
                v = c["sig_beta"] / (n * m)
            elif ck == "sig_eps":
                v = c["sig_eps"] / (n * m * K)
            else:
                fpc = (K_TOT - K) / (K_TOT - 1) if K_TOT > 1 and K < K_TOT else 0.0
                v = c["sig_gamma"] / K * fpc
            vals.append(v)
        vals = np.array(vals)
        ax.bar(x, vals, 0.6, bottom=bottom, label=comp_labels[ic], color=clist[ic])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Benchmark score variance")
    ax.legend(frameon=False, fontsize=7)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    fig.savefig(PLOT_DIR / "decomposition_operating_point.pdf")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'decomposition_operating_point.pdf'}")

# ── Summary ──────────────────────────────────────────────────────────────────

def write_summary(all_comp, all_ci, all_val):
    models = [m for m in MODELS if m in all_comp]
    lines = []
    lines.append("=" * 72)
    lines.append("VARIANCE DECOMPOSITION RESULTS SUMMARY")
    lines.append(f"MT-Bench  |  {len(models)}/{len(MODELS)} models with data  "
                 f"|  {K_TOT} judges × 10 generations")
    lines.append("=" * 72)

    for model in models:
        c = all_comp[model]
        ci = all_ci[model]
        n, m, K = c["n"], c["m"], c["K"]

        lines.append(f"\n{'─' * 72}")
        lines.append(f"Model: {model}")
        lines.append(f"  Scenarios used: {n}/80  |  Generations: {m}  |  Judges: {K}")
        lines.append(f"\n  Estimated true score (μ):  {c['mu']:.4f}  "
                      f"95% CI [{ci['mu'][0]:.4f}, {ci['mu'][1]:.4f}]")

        total_var = predict_var(c["sig_ad"], c["sig_beta"], c["sig_eps"],
                                c["sig_gamma"], n, m, K, K_TOT)
        se = np.sqrt(total_var)
        lines.append(f"  Standard error:            {se:.4f}")
        lines.append(f"  95% CI for μ:              [{c['mu'] - 1.96*se:.4f}, "
                      f"{c['mu'] + 1.96*se:.4f}]")

        lines.append(f"\n  Variance components (raw):")
        for key, label in [("sig_ad", "σ²(α+δ) scenario"), ("sig_beta", "σ²β generation"),
                           ("sig_eps", "σ²ε judge noise"), ("sig_gamma", "σ²γ judge bias")]:
            lines.append(f"    {label:26s} = {c[key]:.6f}  "
                          f"CI [{ci[key][0]:.6f}, {ci[key][1]:.6f}]")

        lines.append(f"\n  Variance decomposition at operating point (n={n}, m={m}, K={K}):")
        parts = [
            ("Scenario σ²(α+δ)/n", c["sig_ad"] / n),
            ("Generation σ²β/(nm)", c["sig_beta"] / (n * m)),
            ("Judge noise σ²ε/(nmK)", c["sig_eps"] / (n * m * K)),
            ("Judge bias (FPC)", c["sig_gamma"] / K * ((K_TOT-K)/(K_TOT-1) if K < K_TOT else 0)),
        ]
        for label, v in parts:
            pct = v / total_var * 100 if total_var > 0 else 0
            lines.append(f"    {label:30s} = {v:.2e}  ({pct:5.1f}%)")

        lines.append(f"\n  Judge biases (γ̂ℓ):")
        for judge in JUDGES:
            lines.append(f"    {JUDGE_LABELS[judge]:12s}: {c['gamma'][judge]:+.4f}")

        if c["sig_gamma"] > 0:
            B_star = (K_TOT - 1) * c["sig_beta"] / c["sig_gamma"]
            lines.append(f"\n  Break-even budget B* = {B_star:.1f}")
            lines.append(f"    → For B > {B_star:.0f}: use all {K_TOT} judges")
            lines.append(f"    → For B < {B_star:.0f}: maximize generations (K=1)")
        else:
            lines.append(f"\n  Break-even budget B* = ∞ (negligible judge bias)")

        # Prediction accuracy (scenario component excluded -- held fixed in subsampling)
        rows = all_val[model]
        preds = np.array([r["predicted"] for r in rows])
        acts = np.array([r["actual"] for r in rows])
        pos = (preds > 0) & (acts > 0)
        if pos.sum() > 2:
            r2 = _r2(preds[pos], acts[pos])
            mape = np.mean(np.abs(preds[pos] - acts[pos]) / acts[pos]) * 100
        else:
            r2, mape = float("nan"), float("nan")
        lines.append(f"\n  D-study validation (excl. scenario component):")
        lines.append(f"    R² (predicted vs actual) = {r2:.4f}")
        lines.append(f"    MAPE                     = {mape:.1f}%")

    txt = "\n".join(lines)
    Path("results_summary.txt").write_text(txt)
    print(txt)
    print(f"\n  Saved results_summary.txt")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    PLOT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    data = load_data()

    all_comp = {}
    all_ci = {}
    all_val = {}

    analyzable_models = []
    for model in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {model}")
        md = data.get(model, {})
        if not md:
            print(f"  SKIPPED — no data yet for {model}")
            continue
        X, qids = build_tensor(md)
        if X.shape[0] == 0:
            print(f"  SKIPPED — no complete scenarios yet (need 10 gens × {K_TOT} judges)")
            continue
        print(f"  Tensor shape: {X.shape}  (n={X.shape[0]}, m={X.shape[1]}, K={X.shape[2]})")

        comp = estimate_components(X)
        all_comp[model] = comp

        print(f"  μ = {comp['mu']:.4f}")
        print(f"  σ²ε={comp['sig_eps']:.4f}  σ²β={comp['sig_beta']:.4f}  "
              f"σ²(α+δ)={comp['sig_ad']:.4f}  σ²γ={comp['sig_gamma']:.4f}")

        print("  Bootstrap CIs (2000 resamples)...")
        ci = bootstrap_ci(X, B=2000)
        all_ci[model] = ci

        print("  Subsampling validation (8000 reps × 25 designs)...")
        val = subsample_validation(X, comp, n_rep=8000)
        all_val[model] = val
        analyzable_models.append(model)

    if not analyzable_models:
        print("\nNo models have enough data for analysis yet.")
        print("Re-run after more generations/judgments have been collected.")
        return

    print(f"\n{'=' * 60}")
    print(f"Generating plots for {len(analyzable_models)}/{len(MODELS)} models...")
    plot_variance_components(all_comp, all_ci)
    plot_judge_biases(all_comp)
    plot_predicted_vs_actual(all_val)
    plot_strategy_comparison(all_comp)
    plot_decomposition_stacked(all_comp)

    print("\nWriting summary...")
    write_summary(all_comp, all_ci, all_val)


if __name__ == "__main__":
    main()
