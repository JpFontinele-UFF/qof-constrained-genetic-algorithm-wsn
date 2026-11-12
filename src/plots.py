"""Figuras dos experimentos (CLAUDE.md, seção 8): PNG, 300 dpi, textos em português.

Cada método tem sempre a mesma cor (a cor segue o método, não a ordem no gráfico).
Grandezas de escalas diferentes ficam em painéis separados (nunca dois eixos y).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

# Paleta categórica (validada para daltonismo) com cor fixa por método
COLORS = {
    "AbII": "#2a78d6",      # azul
    "AG-RQ": "#eb6834",     # laranja
    "AbII+Rep": "#1baf7a",  # verde-água
    "AbII+Pen": "#e87ba4",  # magenta
    "AbI": "#6250d6",       # violeta
    "PE100": "#898781",     # cinza (referência)
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
DPI = 300
WIDTH = 6.3  # polegadas (largura útil de uma página no formato SBC)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.5,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.axisbelow": True,
})

_comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def _finish(fig, axes, path: Path, shared_legend: bool = True) -> None:
    """Vírgula decimal, legenda única acima dos painéis (não cobre dados) e gravação."""
    handles, labels = [], []
    for ax in np.ravel(axes):
        for axis in (ax.xaxis, ax.yaxis):
            # só eixos numéricos lineares; rótulos de texto ficam como estão
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(_comma)
        if shared_legend:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in labels:
                    handles.append(h)
                    labels.append(lab)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    if shared_legend and handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.5)
        top = 1 - 0.35 / fig.get_figheight()
        fig.tight_layout(rect=(0, 0, 1, top))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


ORACLE_LS = (0, (6, 2, 1.5, 2))  # traço-ponto: referências do oráculo (Ef*, PE*)


def _ref_line(ax, y, label, horizontal=True, oracle=False):
    kw = dict(color=INK2, linestyle=ORACLE_LS if oracle else "--", linewidth=1.0, zorder=1)
    (ax.axhline if horizontal else ax.axvline)(y, label=label, **kw)


def _band(ax, series: pd.DataFrame, method: str, col: str, label=None):
    """Média das sementes por sessão com faixa de ±1 desvio."""
    g = series[series.method == method].groupby("t")[col]
    m, s = g.mean(), g.std(ddof=1)
    c = COLORS[method]
    ax.fill_between(m.index, m - s, m + s, color=c, alpha=0.15, linewidth=0)
    ax.plot(m.index, m.values, color=c, label=label or method)


def _warmup(ax, warmup=10):
    ax.axvspan(0, warmup, color=GRID, alpha=0.5, linewidth=0, zorder=0)


# ---------------------------------------------------------------- E1
def plot_validation_abi(series, path):
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.2), sharex=True)
    for ax, col, ylab in [(axes[0], "pe", "PE"), (axes[1], "qof", "QoF (msg/rodada)")]:
        _warmup(ax)
        for m in ("PE100", "AbI"):
            _band(ax, series, m, col, "Sem controle (PE = 100%)" if m == "PE100" else "Abordagem I")
        ax.set_ylabel(ylab)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Abordagem I (N = 100): média de 30 sementes ± 1 desvio")
    axes[1].set_xlabel("Sessão")
    axes[0].legend(loc="lower left", ncol=2)
    _finish(fig, axes, path)


# ---------------------------------------------------------------- E2
def plot_time(series, col, qof_min, path, ref=None, ref_label=None,
              methods=("AbII", "AG-RQ")):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    _warmup(ax)
    for m in methods:
        _band(ax, series, m, col, "Abordagem II" if m == "AbII" else "AG-RQ (proposta)")
    if ref is not None:
        _ref_line(ax, ref, ref_label, oracle=(col == "pe"))
    ax.set_xlabel("Sessão")
    ax.set_ylabel("QoF (msg/rodada)" if col == "qof" else "PE")
    if col == "pe":
        ax.set_ylim(0, 1.05)
    ax.set_xlim(0, series.t.max())
    ax.legend(loc="upper right", ncol=3)
    _finish(fig, ax, path)


def plot_violation_box(runs, methods, path):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    data = [runs.loc[runs.method == m, "viol_rate"].values for m in methods]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.2),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                    markeredgecolor="none"))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.75)
        patch.set_edgecolor(COLORS[m])
    ax.set_xticks(range(1, len(methods) + 1), methods)
    ax.set_ylabel("Sessões com QoF < QoF$_{min}$ (%)")
    ax.grid(axis="x", visible=False)
    _finish(fig, ax, path)


def plot_ef_qof_bars(runs, methods, qof_min, ef_star, path):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    x = np.arange(len(methods))
    for ax, col, ylab, ref, ref_label in [
            (axes[0], "ef_mean", "Ef média", ef_star, "Ef* (oráculo)"),
            (axes[1], "qof_mean", "QoF média (msg/rodada)", qof_min, "QoF$_{min}$")]:
        g = runs.groupby("method")[col]
        mean, sd = g.mean().reindex(methods), g.std(ddof=1).reindex(methods)
        ax.bar(x, mean.values, width=0.65, color=[COLORS[m] for m in methods],
               yerr=sd.values, error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=2))
        _ref_line(ax, ref, ref_label, oracle=(col == "ef_mean"))
        ax.set_xticks(x, methods, rotation=35, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    axes[0].set_ylim(0, max(ef_star * 1.25, axes[0].get_ylim()[1]))
    _finish(fig, axes, path)


# ---------------------------------------------------------------- E3 / E4
def _errline(ax, x, summ, method, col, shift=lambda v, k: v):
    """Média ± desvio; `shift` desloca levemente o x de cada método (k = -1 ou +1)."""
    g = summ[summ.method == method]
    k = -1 if method == "AbII" else 1
    ax.errorbar(shift(g[x].to_numpy(float), k), g[f"{col}_mean"], yerr=g[f"{col}_sd"], color=COLORS[method],
                marker="o", markersize=4, capsize=2, elinewidth=0.8,
                label="Abordagem II" if method == "AbII" else "AG-RQ (proposta)")


def _shift_lin(v, k):
    return v + 0.08 * k


def _shift_log(v, k):
    return v * 2 ** (0.04 * k)


def plot_threshold(summ, path):
    from .oracle import oracle
    ns = sorted(summ.N.unique())
    fig, axes = plt.subplots(2, len(ns), figsize=(WIDTH, 4.6), sharex=True)
    q = np.array(sorted(summ.qof_thr.unique()))
    for j, n in enumerate(ns):
        s = summ[summ.N == n]
        for m in ("AbII", "AG-RQ"):
            _errline(axes[0, j], "qof_thr", s, m, "qof_mean", _shift_lin)
            _errline(axes[1, j], "qof_thr", s, m, "ef_mean", _shift_lin)
        axes[0, j].plot(q, q, color=INK2, linestyle="--", linewidth=1.0, label="QoF = QoF$_{min}$")
        axes[1, j].plot(q, [oracle(n, v).ef_star for v in q], color=INK2, linestyle=ORACLE_LS,
                        linewidth=1.0, label="Ef* (oráculo)")
        axes[0, j].set_title(f"N = {n}")
        axes[1, j].set_xlabel("QoF$_{min}$")
    axes[0, 0].set_ylabel("QoF média (msg/rodada)")
    axes[1, 0].set_ylabel("Ef média")
    axes[0, 0].legend(loc="upper left")
    axes[1, 0].legend(loc="lower left")
    _finish(fig, axes, path)


def plot_density(summ, qof_min, path):
    from .oracle import oracle
    ns = np.array(sorted(summ.N.unique()))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    for m in ("AbII", "AG-RQ"):
        _errline(axes[0], "N", summ, m, "qof_mean", _shift_log)
        _errline(axes[1], "N", summ, m, "ef_mean", _shift_log)
    _ref_line(axes[0], qof_min, "QoF$_{min}$")
    axes[1].plot(ns, [oracle(int(n), qof_min).ef_star for n in ns], color=INK2,
                 linestyle=ORACLE_LS, linewidth=1.0, label="Ef* (oráculo)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlabel("Número de sensores N")
        ax.minorticks_off()
    axes[0].set_ylabel("QoF média (msg/rodada)")
    axes[1].set_ylabel("Ef média")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")
    _finish(fig, axes, path)


# ---------------------------------------------------------------- E5
def plot_dynamic(series, qof_min, pe_star, switch, path):
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.4), sharex=True)
    for ax, col, ylab in [(axes[0], "qof", "QoF (msg/rodada)"), (axes[1], "pe", "PE")]:
        _warmup(ax)
        for m in ("AbII", "AG-RQ"):
            _band(ax, series, m, col, "Abordagem II" if m == "AbII" else "AG-RQ (proposta)")
        ax.axvline(switch, color=MUTED, linewidth=1.0, linestyle=":")
        ax.set_ylabel(ylab)
    _ref_line(axes[0], qof_min, "QoF$_{min}$")
    axes[1].step(np.arange(len(pe_star)), pe_star, where="post", color=INK2, linestyle=ORACLE_LS,
                 linewidth=1.0, label="PE* (oráculo)")
    axes[0].text(switch + 1, axes[0].get_ylim()[1] * 0.95, "N: 100 → 200", color=INK2,
                 fontsize=8, va="top")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Sessão")
    axes[0].legend(loc="lower left", ncol=3)
    axes[1].legend(loc="upper right", ncol=3)
    _finish(fig, axes, path)


# ---------------------------------------------------------------- E6
# deslocamento (pontos) do rótulo de cada configuração (margem, passo): evita sobreposição
_TRADEOFF_OFFSETS = {(0.0, 0.03): (10, 10), (0.0, 0.06): (14, 0), (0.05, 0.03): (-12, 12),
                     (0.05, 0.06): (14, -10), (0.10, 0.03): (-16, -2), (0.10, 0.06): (12, -10)}


def plot_tradeoff(summ, path):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    rg = summ[summ.method == "AG-RQ"]
    for _, r in rg.iterrows():
        ax.errorbar(r.viol_rate_mean, r.ef_mean_mean, xerr=r.viol_rate_sd, yerr=r.ef_mean_sd,
                    fmt="none", ecolor=COLORS["AG-RQ"], elinewidth=0.6, alpha=0.4)
    ax.scatter(rg.viol_rate_mean, rg.ef_mean_mean, s=36, color=COLORS["AG-RQ"],
               edgecolor="white", linewidth=1.0, zorder=3, label="AG-RQ (margem m, passo δr)")
    off = rg[rg.oficial]
    ax.scatter(off.viol_rate_mean, off.ef_mean_mean, s=110, facecolor="none",
               edgecolor=INK, linewidth=1.0, zorder=4, label="Configuração oficial")
    for _, r in rg.iterrows():
        txt = f"m={r.margin:.2f}; δr={r.repair_step:.2f}".replace(".", ",")
        dx, dy = _TRADEOFF_OFFSETS.get((round(r.margin, 2), round(r.repair_step, 2)), (8, 4))
        ax.annotate(txt, (r.viol_rate_mean, r.ef_mean_mean), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, color=INK2, ha="left" if dx > 0 else "right",
                    va="center", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5,
                                                 shrinkA=0, shrinkB=3))
    names = {"PE100": "Sem controle (PE100)", "AbII": "Abordagem II", "AbII+Pen": "AbII+Pen"}
    ref_offsets = {"PE100": (8, -8), "AbII": (10, -14), "AbII+Pen": (10, 12)}
    for m, lab in names.items():
        r = summ[summ.method == m].iloc[0]
        ax.errorbar(r.viol_rate_mean, r.ef_mean_mean, xerr=r.viol_rate_sd, yerr=r.ef_mean_sd,
                    fmt="none", ecolor=COLORS[m], elinewidth=0.6, alpha=0.4)
        ax.scatter(r.viol_rate_mean, r.ef_mean_mean, s=40, marker="s", color=COLORS[m],
                   edgecolor="white", linewidth=1.0, zorder=3, label=lab)
        ax.annotate(lab, (r.viol_rate_mean, r.ef_mean_mean), textcoords="offset points",
                    xytext=ref_offsets[m], fontsize=7, color=INK2, va="center")
    ax.set_xlabel("Sessões com QoF < QoF$_{min}$ (%)")
    ax.set_ylabel("Ef média")
    ax.set_xlim(left=-2)
    ax.legend(loc="lower right", fontsize=7)
    _finish(fig, ax, path, shared_legend=False)
