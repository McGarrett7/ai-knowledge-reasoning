"""Figures for the manuscript (PNG, 300 dpi, light background for print)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import FIGURES_DIR  # noqa: E402

# categorical slots in fixed order (validated: CVD-separated, labels added for contrast)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ALGS = ["Algorithm 1 (paraconsistent)", "Algorithm 2 (learning-assisted)", "Algorithm 3 (multi-agent)"]
SHORT = {a: f"Algorithm {i + 1}" for i, a in enumerate(ALGS)}


def _style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return path


def performance(summary: dict, name="reasoning_performance.png"):
    """Fig. 1/2: correctness, conflict mitigation, accuracy, F1 (one 0-1 axis) and
    support ratio on its own panel (different scale)."""
    metrics = [("correctness", "Correctness"), ("conflict_mitigation", "Conflict mitigation"),
               ("accuracy", "Accuracy"), ("f1", "F1")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(ALGS))
    w = 0.2
    for i, (m, label) in enumerate(metrics):
        mean = [summary[a][m]["mean"] for a in ALGS]
        std = [summary[a][m]["std"] for a in ALGS]
        bars = ax.bar(x + (i - 1.5) * w, mean, w, yerr=std, color=SERIES[i], label=label,
                      edgecolor=SURFACE, linewidth=1.5, error_kw={"ecolor": MUTED, "lw": 0.8, "capsize": 2})
        for b, v in zip(bars, mean):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK, rotation=90)
    ax.set_xticks(x, [SHORT[a] for a in ALGS])
    lo = min(summary[a][m]["mean"] for a in ALGS for m, _ in metrics)
    ax.set_ylim(max(0.0, lo - 0.1), 1.08)
    ax.set_ylabel("Score (test set, mean ± std)", color=INK)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    _style(ax)
    sr = [summary[a]["support_ratio"]["mean"] for a in ALGS]
    bars = ax2.bar(x, sr, 0.5, color=MUTED, edgecolor=SURFACE)
    for b, v in zip(bars, sr):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax2.set_xticks(x, ["Alg. 1", "Alg. 2", "Alg. 3"])
    ax2.set_title("Support ratio", fontsize=9, color=INK)
    _style(ax2)
    return _save(fig, name)


def robustness(clean: dict, noisy: dict, name="noise_robustness.png"):
    """Fig. 3: correctness on the clean vs. noisy KG."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, a in enumerate(ALGS):
        ys = [clean[a]["correctness"]["mean"], noisy[a]["correctness"]["mean"]]
        ax.plot([0, 1], ys, color=SERIES[i], lw=2, marker="o", ms=7, label=SHORT[a])
        ax.annotate(f"{SHORT[a]}  {ys[1]:.3f}", (1, ys[1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1], ["Clean KG", "Noisy KG"])
    ax.set_xlim(-0.2, 1.6)
    ax.set_ylabel("Correctness", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    return _save(fig, name)


def training(history: list, name="training_dynamics.png"):
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    a1.plot(ep, [h["loss"] for h in history], color=SERIES[0], lw=2)
    a1.set_title("Training loss", fontsize=9, color=INK)
    a2.plot(ep, [h["val_acc"] for h in history], color=SERIES[0], lw=2)
    a2.set_title("Validation accuracy", fontsize=9, color=INK)
    for a in (a1, a2):
        a.set_xlabel("Epoch", color=INK)
        _style(a)
    return _save(fig, name)


def sota(summary: dict, methods: list, name="sota_comparison.png"):
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(methods) + 1.2))
    y = np.arange(len(methods))
    for i, (m, label) in enumerate([("accuracy", "Accuracy"), ("f1", "F1")]):
        vals = [summary[k][m]["mean"] for k in methods]
        err = [summary[k][m]["std"] for k in methods]
        bars = ax.barh(y + (i - 0.5) * 0.38, vals, 0.38, xerr=err, color=SERIES[i], label=label,
                       edgecolor=SURFACE, error_kw={"ecolor": MUTED, "lw": 0.8})
        for b, v in zip(bars, vals):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, methods, fontsize=8)
    ax.invert_yaxis()
    lo = min(summary[k][m]["mean"] for k in methods for m in ("accuracy", "f1"))
    ax.set_xlim(max(0.0, lo - 0.05), 1.06)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _style(ax, ygrid=False)
    return _save(fig, name)


def scalability(result: dict, name="scalability.png"):
    rows = result["size"]
    k = [r["triples"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    cols = [("fragments_s", "Fragment construction"), ("train_s_per_epoch", "Training, 1 epoch (sampled)"),
            ("alg2_inference_s", "Alg. 2 inference"), ("alg3_inference_s", "Alg. 3 inference")]
    for i, (c, label) in enumerate(cols):
        slope = result["loglog_slope"][c]
        a1.plot(k, [max(r[c], 1e-4) for r in rows], color=SERIES[i], lw=2, marker="o", ms=5,
                label=f"{label} (slope {slope:.2f})")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("Triples |K|", color=INK)
    a1.set_ylabel("Seconds", color=INK)
    a1.legend(frameon=False, fontsize=7)
    _style(a1)
    ag = result["agents"]
    a2.plot([r["agents"] for r in ag], [r["train_s_per_epoch"] for r in ag], color=SERIES[0], lw=2,
            marker="o", label="Local training, 1 epoch (all agents)")
    a2.plot([r["agents"] for r in ag], [r["inference_s"] for r in ag], color=SERIES[1], lw=2,
            marker="o", label="Aggregation + reasoning")
    a2.set_xlabel("Number of agents m", color=INK)
    a2.set_ylabel("Seconds", color=INK)
    a2.legend(frameon=False, fontsize=7)
    _style(a2)
    return _save(fig, name)


def analysis_robustness(result: dict, name="scorer_robustness.png"):
    """Scorer choice inside Algorithm 2 under more noise and fewer training triples."""
    from experiments.analysis import GNN_ONLY, SCORERS
    methods = [s[0] for s in SCORERS] + [GNN_ONLY]
    short = ["ID embedding", "Feature MLP", "R-GCN (proposed)", "R-GCN, no paraconsistent"]
    noise = [s for s in result["settings"] if s["kind"] == "noise"]
    frac = [s for s in result["settings"] if s["kind"] == "noise" and s["value"] == noise[0]["value"]] + \
        [s for s in result["settings"] if s["kind"] == "train_fraction"]
    panels = [(noise, "accuracy", "Accuracy", "Injected noise triples", lambda s: s["noise_triples"]),
              (noise, "conflict_resolution_acc", "Conflict resolution acc.", "Injected noise triples",
               lambda s: s["noise_triples"]),
              (frac, "accuracy", "Accuracy", "Fraction of training triples", lambda s: s["train_fraction"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (settings, key, ylab, xlab, xf) in zip(axes, panels):
        xs = [xf(s) for s in settings]
        for i, m in enumerate(methods):
            ys = [s["metrics"][m][key]["mean"] for s in settings]
            ax.plot(xs, ys, color=SERIES[i], lw=2, marker="osD^"[i], ms=(9, 7, 5, 6)[i],
                    markeredgecolor=SURFACE, markeredgewidth=1, label=short[i], zorder=3 + i)
        ax.set_xticks(xs)
        ax.set_xlabel(xlab, color=INK)
        ax.set_ylabel(ylab, color=INK)
        if xlab.startswith("Fraction"):
            ax.invert_xaxis()
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, name)
