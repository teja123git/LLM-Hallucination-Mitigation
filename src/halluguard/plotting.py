"""All figures. Every function returns a matplotlib Figure and, when ``path`` is given, also saves it (PNG, 200 dpi).
Figures made in mock mode carry a diagonal 'MOCK DATA' watermark."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

if not matplotlib.get_backend().lower().startswith(("module://", "nbagg", "tkagg", "macosx", "qt")):
    try:
        matplotlib.use("Agg")
    except Exception:  # pragma: no cover
        pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold", "axes.titlesize": 11})

MOCK = {"on": False}
SYS_COLORS = {"A": "#9aa5b1", "B": "#4c78a8", "C": "#f58518", "D": "#e45756", "E": "#72b7b2", "F": "#54a24b", "G": "#b279a2", "F_cove": "#9d755d"}
SYS_LABELS = {"A": "A  LLM only", "B": "B  +RAG", "C": "C  +self-consistency", "D": "D  +CoVe verification", "E": "E  +RAG+verification",
              "F": "F  +correction (RARR)", "G": "G  hybrid+correction", "F_cove": "F-CoVe  +correction (CoVe)"}
DET_COLORS = {"nli": "#4c78a8", "llm": "#f58518", "selfcheck": "#e45756", "cove": "#9d755d", "evidence": "#54a24b", "hybrid": "#b279a2",
              "nli_gold": "#a0cbe8", "llm_gold": "#ffbe7d"}


def set_mock(flag: bool) -> None:
    MOCK["on"] = bool(flag)


def _finish(fig, path: Optional[str | Path], tight: bool = True):
    if tight:
        fig.tight_layout()
    if MOCK["on"]:
        fig.text(0.5, 0.5, "MOCK DATA\nnot real results", fontsize=38, color="red", alpha=0.18, ha="center", va="center", rotation=28, weight="bold")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    if matplotlib.get_backend().lower() == "agg":   # headless (CLI / tests): free the memory; notebooks keep figures for inline display
        plt.close(fig)
    return fig


def _sys_order(df: pd.DataFrame, col: str = "system") -> List[str]:
    order = list(SYS_COLORS)
    return [s for s in order if s in set(df[col])]


# ============================================================================ Stage 1
def fig_eda(tqa: pd.DataFrame, halu: pd.DataFrame, path=None):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    vc = tqa["category"].value_counts().head(10)
    sns.barplot(x=vc.values, y=vc.index, ax=ax[0], hue=vc.index, palette="crest", legend=False)
    ax[0].set(title="TruthfulQA: top categories (sample)", xlabel="questions", ylabel="")
    ax[1].hist(halu["right_answer"].str.split().str.len(), bins=20, alpha=.65, label="right answer", color="#4ecdc4", edgecolor="k")
    ax[1].hist(halu["hallucinated_answer"].str.split().str.len(), bins=20, alpha=.65, label="hallucinated answer", color="#ff6b6b", edgecolor="k")
    ax[1].set(title="HaluEval: answer length (words)", xlabel="words", ylabel="count")
    ax[1].legend()
    ax[2].hist(halu["knowledge"].str.split().str.len(), bins=25, color="#6c8ebf", edgecolor="k")
    ax[2].set(title="HaluEval: gold paragraph length (words)", xlabel="words", ylabel="count")
    return _finish(fig, path)


def fig_retrieval(df: pd.DataFrame, path=None):
    """df columns: dataset, mode, recall@1/3/5/10, mrr"""
    ks = [c for c in df.columns if c.startswith("recall@")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for (ds, mode), g in df.groupby(["dataset", "mode"]):
        ax[0].plot([int(k.split("@")[1]) for k in ks], g[ks].iloc[0].values, marker="o", label=f"{ds} / {mode}", linestyle="-" if ds == "halueval" else "--")
    ax[0].set(title="Retrieval Recall@k (gold passage in top-k)", xlabel="k", ylabel="recall", ylim=(0, 1.02))
    ax[0].legend(fontsize=8)
    sns.barplot(data=df, x="dataset", y="mrr", hue="mode", ax=ax[1])
    ax[1].set(title="Mean reciprocal rank", ylim=(0, 1.02))
    return _finish(fig, path)


def fig_stage1_quality(df: pd.DataFrame, path=None):
    """df: per-question rows for systems A/B with dataset column."""
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
    order = _sys_order(df)
    pal = {s: SYS_COLORS[s] for s in order}
    first = "bertscore_f1" if "bertscore_f1" in df else "token_f1" if "token_f1" in df else "answer_words"
    titles = {"bertscore_f1": "BERTScore F1 vs reference", "token_f1": "Token F1 vs reference (HaluEval)", "answer_words": "Answer length (words)"}
    for j, col, title in ((0, first, titles[first]), (1, "rougeL_f1", "ROUGE-L F1 vs reference")):
        sns.boxplot(data=df, x="dataset", y=col, hue="system", hue_order=order, palette=pal, ax=ax[0, j])
        ax[0, j].set(title=title, xlabel="")
    acc = []
    for _, r in df.iterrows():
        if r["dataset"] == "halueval":
            acc.append({"dataset": "HaluEval\n(gold answer contained)", "system": r["system"], "value": r.get("contains", np.nan)})
        elif not r["abstained"]:
            acc.append({"dataset": "TruthfulQA\n(truthful by similarity,\nattempted only)", "system": r["system"], "value": r.get("tqa_truthful_sim", np.nan)})
    a = pd.DataFrame(acc)
    if len(a):
        sns.barplot(data=a, x="dataset", y="value", hue="system", hue_order=order, palette=pal, ax=ax[0, 2], errorbar=("ci", 95))
        ax[0, 2].set(title="Task accuracy (95% CI)", xlabel="", ylabel="rate", ylim=(0, 1.05))
    ab = df.groupby(["dataset", "system"])["abstained"].mean().reset_index()
    sns.barplot(data=ab, x="dataset", y="abstained", hue="system", hue_order=order, palette=pal, ax=ax[1, 0])
    ax[1, 0].set(title="Abstention rate", ylabel="fraction", xlabel="", ylim=(0, 1.05))
    for j, (col, title) in enumerate((("cost_latency", "Latency per question (s)"), ("cost_total_tokens", "Tokens per question"))):
        if col in df:
            sns.barplot(data=df, x="dataset", y=col, hue="system", hue_order=order, palette=pal, ax=ax[1, j + 1], errorbar=("ci", 95))
            ax[1, j + 1].set(title=title, xlabel="")
    for a_ in ax.ravel():
        leg = a_.get_legend()
        if leg:
            leg.set_title("")
    return _finish(fig, path)


# ============================================================================ Stage 2
def fig_roc_pr(curves: Dict[str, tuple], title: str, path=None):
    """curves: name -> (y_true, score)"""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    pos = None
    for name, (y, s) in curves.items():
        y, s = np.asarray(y), np.asarray(s)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        c = DET_COLORS.get(name)
        ax[0].plot(fpr, tpr, label=f"{name} (AUROC {auc(fpr, tpr):.3f})", color=c)
        ax[1].plot(r, p, label=f"{name} (AUPRC {auc(r, p):.3f})", color=c)
        pos = y.mean()
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(title=f"ROC - {title}", xlabel="false-positive rate", ylabel="true-positive rate")
    if pos is not None:
        ax[1].axhline(pos, color="k", ls=":", lw=1)
    ax[1].set(title=f"Precision-recall - {title}", xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    return _finish(fig, path)


def fig_detector_bars(df: pd.DataFrame, metrics: Sequence[str] = ("auroc", "auprc", "f1"), path=None):
    """df: dataset, detector, <metrics>"""
    fig, ax = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.6), squeeze=False)
    for j, m in enumerate(metrics):
        sns.barplot(data=df, x="dataset", y=m, hue="detector", palette=DET_COLORS, ax=ax[0, j])
        ax[0, j].set(title=m.upper(), xlabel="", ylim=(0, 1.02))
        ax[0, j].get_legend().set_title("") if j == len(metrics) - 1 else ax[0, j].get_legend().remove()
    return _finish(fig, path)


def fig_score_hist(scores: Dict[str, tuple], path=None):
    fig, ax = plt.subplots(1, len(scores), figsize=(4.3 * len(scores), 3.8), squeeze=False)
    for j, (name, (y, s)) in enumerate(scores.items()):
        y, s = np.asarray(y), np.asarray(s)
        ax[0, j].hist(s[y == 0], bins=20, range=(0, 1), alpha=.65, label="not hallucinated", color="#4ecdc4", edgecolor="k")
        ax[0, j].hist(s[y == 1], bins=20, range=(0, 1), alpha=.65, label="hallucinated", color="#ff6b6b", edgecolor="k")
        ax[0, j].set(title=name, xlabel="detector score")
    ax[0, 0].legend(fontsize=8)
    return _finish(fig, path)


def fig_confusions(cms: Dict[str, np.ndarray], labels: Sequence[str], path=None, normalize: bool = True):
    fig, ax = plt.subplots(1, len(cms), figsize=(4.4 * len(cms), 4), squeeze=False)
    for j, (name, cm) in enumerate(cms.items()):
        cm = np.asarray(cm, float)
        shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
        sns.heatmap(shown, annot=cm.astype(int) if not normalize else np.round(shown, 2), fmt="" if normalize else "d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax[0, j])
        ax[0, j].set(title=name, xlabel="predicted", ylabel="true")
    return _finish(fig, path)


def fig_lines(df: pd.DataFrame, x: str, y: str, hue: str, title: str, path=None, xlabel: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o", ax=ax)
    ax.set(title=title, xlabel=xlabel or x, ylim=(0, 1.02))
    return _finish(fig, path)


# ============================================================================ Stage 3
def fig_grades(summary: pd.DataFrame, path=None):
    """Stacked correct / not-attempted / incorrect (hallucination) share per system, one panel per dataset."""
    dsets = list(summary["dataset"].unique())
    fig, ax = plt.subplots(1, len(dsets), figsize=(6.8 * len(dsets), 4.8), squeeze=False)
    for j, ds in enumerate(dsets):
        s = summary[summary["dataset"] == ds].set_index("system").reindex(_sys_order(summary))
        s = s.dropna(subset=["correct"])
        labels = [SYS_LABELS[i] for i in s.index]
        ax[0, j].barh(labels, s["correct"], color="#54a24b", label="correct")
        ax[0, j].barh(labels, s["not_attempted"], left=s["correct"], color="#c9ced6", label="not attempted (abstained)")
        ax[0, j].barh(labels, s["incorrect"], left=s["correct"] + s["not_attempted"], color="#e45756", label="incorrect (hallucination)")
        ax[0, j].invert_yaxis()
        ax[0, j].set(title=f"Judge grades - {ds}", xlim=(0, 1))
    ax[0, 0].legend(loc="lower left", fontsize=8, bbox_to_anchor=(0, -0.28), ncol=3)
    return _finish(fig, path)


def fig_metric_ci(summary: pd.DataFrame, metric: str, lo: str, hi: str, title: str, path=None):
    dsets = list(summary["dataset"].unique())
    fig, ax = plt.subplots(1, len(dsets), figsize=(6 * len(dsets), 4.4), squeeze=False)
    for j, ds in enumerate(dsets):
        s = summary[summary["dataset"] == ds].set_index("system").reindex(_sys_order(summary)).dropna(subset=[metric])
        err = np.vstack([s[metric] - s[lo], s[hi] - s[metric]]).clip(min=0)
        ax[0, j].bar([SYS_LABELS[i].split("  ")[0] for i in s.index], s[metric], yerr=err, capsize=4, color=[SYS_COLORS[i] for i in s.index], edgecolor="k")
        ax[0, j].set(title=f"{title} - {ds}", ylim=(0, 1.05))
    return _finish(fig, path)


def fig_cost(summary_all: pd.DataFrame, group_costs: pd.DataFrame, path=None):
    """summary_all: system, cost_latency, cost_total_tokens, cost_calls;  group_costs: system, group, total_tokens, latency"""
    order = _sys_order(summary_all)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    groups = list(dict.fromkeys(group_costs["group"]))
    pal = dict(zip(groups, sns.color_palette("Set2", len(groups))))
    for a, col, title in ((ax[0], "total_tokens", "Tokens per question by stage group"), (ax[1], "latency", "Latency per question by stage group (s)")):
        bottom = np.zeros(len(order))
        for g in groups:
            vals = np.array([group_costs[(group_costs["system"] == s) & (group_costs["group"] == g)][col].sum() for s in order])
            a.bar([s for s in order], vals, bottom=bottom, label=g, color=pal[g], edgecolor="k", lw=.4)
            bottom += vals
        a.set(title=title)
    ax[1].legend(fontsize=8)
    s = summary_all.set_index("system").reindex(order)
    ax[2].bar(order, s["cost_calls"], color=[SYS_COLORS[i] for i in order], edgecolor="k")
    ax[2].set(title="LLM calls per question")
    return _finish(fig, path)


def fig_pareto(summary_all: pd.DataFrame, y: str = "f_score", ylabel: str = "SimpleQA F-score (higher = better)", path=None):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    s = summary_all.dropna(subset=[y])
    for a, x, xl in ((ax[0], "cost_total_tokens", "tokens per question"), (ax[1], "cost_latency", "latency per question (s)")):
        for _, r in s.iterrows():
            a.scatter(r[x], r[y], s=110, color=SYS_COLORS[r["system"]], edgecolor="k", zorder=3)
            a.annotate(r["system"], (r[x], r[y]), textcoords="offset points", xytext=(6, 5), fontsize=10)
        a.set(xlabel=xl, ylabel=ylabel, title="Factuality vs. cost")
    return _finish(fig, path)


def fig_marginal(contrib: pd.DataFrame, path=None):
    """contrib: contrast, delta_correct, lo, hi, delta_incorrect (paired differences, per dataset optional)"""
    fig, ax = plt.subplots(1, 2, figsize=(13, max(3.8, 0.55 * len(contrib["contrast"].unique()) + 1.5)))
    for a, col, lo, hi, title, good in ((ax[0], "delta_correct", "correct_lo", "correct_hi", "Change in CORRECT rate", True),
                                         (ax[1], "delta_incorrect", "incorrect_lo", "incorrect_hi", "Change in INCORRECT (hallucination) rate", False)):
        c = contrib.dropna(subset=[col])
        y = np.arange(len(c))
        err = np.vstack([c[col] - c[lo], c[hi] - c[col]]).clip(min=0)
        colors = [("#54a24b" if (v > 0) == good else "#e45756") if abs(v) > 1e-9 else "#999" for v in c[col]]
        a.barh(y, c[col], xerr=err, color=colors, edgecolor="k", capsize=3)
        a.set_yticks(y)
        a.set_yticklabels(c["contrast"], fontsize=9)
        a.axvline(0, color="k", lw=1)
        a.invert_yaxis()
        a.set(title=title, xlabel="paired difference (95% bootstrap CI)")
    return _finish(fig, path)


def fig_claim_rates(summary: pd.DataFrame, path=None):
    s = summary.dropna(subset=["claim_support_rate"])
    order = _sys_order(s)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(order))
    w = .38
    t = s.groupby("system")[["claim_support_rate", "claim_contradiction_rate"]].mean().reindex(order)
    ax.bar(x - w / 2, t["claim_support_rate"], w, label="supported (FActScore-style precision)", color="#54a24b", edgecolor="k")
    ax.bar(x + w / 2, t["claim_contradiction_rate"], w, label="contradicted", color="#e45756", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set(title="Claim-level factuality of the final answers (judge vs. gold references)", ylim=(0, 1.05))
    ax.legend(fontsize=8)
    return _finish(fig, path)


def fig_heatmap(summary: pd.DataFrame, metrics: Sequence[str], path=None):
    cols = [m for m in metrics if m in summary]
    t = summary.pivot_table(index="system", columns="dataset", values=cols)
    t = t.reindex(_sys_order(summary))
    fig, ax = plt.subplots(figsize=(1.9 * t.shape[1] + 2, 0.55 * len(t) + 1.8))
    sns.heatmap(t, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar=False)
    ax.set(title="Metric overview", xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=30)
    return _finish(fig, path)


def fig_flag_stats(df: pd.DataFrame, path=None):
    g = df[df["system"].isin(["C", "D", "E", "F", "G", "F_cove"])].groupby("system")[["n_flagged", "n_removed", "n_rewritten"]].mean()
    g = g.reindex([s for s in SYS_COLORS if s in g.index])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    g.plot.bar(ax=ax, color=["#f58518", "#e45756", "#54a24b"], edgecolor="k")
    ax.set(title="Claims flagged / removed / rewritten per question", ylabel="mean count", xlabel="")
    ax.tick_params(axis="x", rotation=0)
    return _finish(fig, path)


def fig_threshold_retention(scores: Dict[str, Sequence[float]], thresholds: Dict[str, float], path=None):
    """Fraction of claims that survive filtering as a function of the detector threshold (a free sensitivity analysis)."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    grid = np.linspace(0, 1, 101)
    for name, sc in scores.items():
        sc = np.asarray([x for x in sc if x is not None], float)
        if not len(sc):
            continue
        ax.plot(grid, [(sc < t).mean() for t in grid], label=name, color=DET_COLORS.get(name))
        if name in thresholds:
            ax.axvline(thresholds[name], color=DET_COLORS.get(name), ls=":", lw=1)
    ax.set(title="Claims retained vs. detector threshold (dotted = calibrated)", xlabel="threshold (flag if score >= t)", ylabel="fraction of claims kept", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    return _finish(fig, path)
