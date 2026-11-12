"""Experimentos E1–E6 (CLAUDE.md, seção 8).

Uso: python -m src.experiments --exp E2      (ou E1..E6, all)
     python -m src.experiments --exp all --seeds 5   (rodada rápida)
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import plots
from .config import C1Config, SimConfig
from .controllers import PE100, AbI, AbII, Proposed
from .metrics import METRIC_COLS, compute_metrics
from .oracle import oracle
from .simulator import dynamic_schedule, run

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figuras"
N_SEEDS = 30

METHODS = {
    "PE100": lambda c1: PE100(),
    "AbI": lambda c1: AbI(),
    "AbII": lambda c1: AbII(),
    "AbII+Pen": lambda c1: Proposed(True, False, c1),
    "AbII+Rep": lambda c1: Proposed(False, True, c1),
    "AG-RQ": lambda c1: Proposed(True, True, c1),
}
VARIANTS = ["AbII+Pen", "AbII+Rep", "AG-RQ"]
SERIES_COLS = ["t", "n_nodes", "interf", "pe", "sent", "received", "ef", "qof",
               "dpe_rule", "dpe_applied", "repaired"]
TABLE_A2 = {25: (12.0, 0.49), 50: (16.0, 0.322), 100: (19.0, 0.19), 200: (21.0, 0.105)}
BLOCKS = [(10, 30), (30, 50), (50, 70), (70, 100)]

# Rótulos em português para tabelas
LABELS = {"ef_mean": "Ef média", "qof_mean": "QoF média", "qof_min": "QoF mínima",
          "viol_rate": "Violação (%)", "pe_mean": "PE média", "pe_std": "Desvio da PE",
          "pe_reversals": "Inversões da PE", "sent_total": "Total enviado",
          "repairs": "Reparos", "gap_oracle": "Gap ao oráculo"}
DIGITS = {"ef_mean": 3, "qof_mean": 2, "qof_min": 2, "viol_rate": 1, "pe_mean": 3,
          "pe_std": 3, "pe_reversals": 1, "sent_total": 0, "repairs": 1, "gap_oracle": 3}


# ------------------------------------------------------------------ execução
def run_batch(exp: str, method: str, seeds, n_nodes=100, qof_min: float = 12.0,
              c1: C1Config | None = None, extra: dict | None = None,
              segments: dict | None = None, p_interf: float | None = None):
    """Roda um método em várias sementes; devolve (métricas, séries)."""
    sim = replace(SimConfig(), qof_min=qof_min)
    if p_interf is not None:
        sim = replace(sim, channel=replace(sim.channel, p_interf=p_interf))
    n_label = n_nodes if np.isscalar(n_nodes) else "100->200"
    ids = {"exp": exp, "method": method, "N": n_label, "qof_thr": qof_min, **(extra or {})}
    rows, series = [], []
    for seed in seeds:
        df = run(METHODS[method](c1), seed=seed, n_nodes=n_nodes, sim_cfg=sim)
        rows.append({**ids, "seed": seed, "segment": "total",
                     **compute_metrics(df, qof_min, sim.warmup)})
        for seg, (a, b) in (segments or {}).items():
            rows.append({**ids, "seed": seed, "segment": seg,
                         **compute_metrics(df, qof_min, a, b)})
        s = df[SERIES_COLS].copy()
        for k, v in {**ids, "seed": seed}.items():
            s.insert(0, k, v)
        series.append(s)
    return pd.DataFrame(rows), pd.concat(series, ignore_index=True)


def collect(batches):
    runs, series = zip(*batches)
    return pd.concat(runs, ignore_index=True), pd.concat(series, ignore_index=True)


# ------------------------------------------------------------------- tabelas
def summarize(runs: pd.DataFrame, by: list[str], metrics=METRIC_COLS) -> pd.DataFrame:
    """Média e desvio (entre sementes) de cada métrica, por grupo."""
    g = runs.groupby(by, sort=False)[list(metrics)]
    mean, sd = g.mean(), g.std(ddof=1)
    out = pd.concat({"mean": mean, "sd": sd}, axis=1)
    out.columns = [f"{m}_{stat}" for stat, m in out.columns]
    return out.reset_index()


def fmt_ms(mean: float, sd: float, digits: int) -> str:
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def mean_sd_table(summary: pd.DataFrame, keys: list[str], metrics) -> pd.DataFrame:
    """Tabela de texto 'média (desvio)' para exibição."""
    out = summary[keys].copy()
    for m in metrics:
        out[LABELS.get(m, m)] = [fmt_ms(a, b, DIGITS.get(m, 3))
                                 for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_sd"])]
    return out


def _tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("->", r"$\to$"), ("→", r"$\to$")]:
        s = s.replace(a, b)
    return s


def _tex_number(s: str) -> str:
    """Vírgula decimal no LaTeX: 0.248 -> 0{,}248."""
    import re
    return re.sub(r"(\d)\.(\d)", r"\1{,}\2", s)


def write_tex(df: pd.DataFrame, path: Path) -> None:
    """Tabela booktabs (sem depender do jinja2)."""
    align = "l" * 1 + "r" * (df.shape[1] - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(_tex_escape(str(c)) for c in df.columns) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(_tex_number(_tex_escape(str(v))) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_table(df: pd.DataFrame, name: str, out: Path) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    write_tex(df, out / f"{name}.tex")


def wilcoxon_p(x, y) -> float:
    """Wilcoxon pareado bicaudal; p = 1 se todas as diferenças forem nulas."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(x, y).pvalue)


def block_table(series: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """PE e QoF médias por trecho de sessões (média sobre sementes e sessões)."""
    rows = []
    for key, g in series.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for a, b in BLOCKS:
            seg = g[(g.t >= a) & (g.t < b)]
            row[f"PE {a}-{b - 1}"] = round(seg.pe.mean(), 3)
            row[f"QoF {a}-{b - 1}"] = round(seg.qof.mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experimentos
def exp_e1(seeds, out, fig):
    batches = []
    for n in TABLE_A2:
        # sem interferência: verificação da calibração pela Tabela A.2
        batches.append(run_batch("E1", "PE100", seeds, n, p_interf=0.0,
                                 extra={"cenario": "sem interferencia"}))
        batches.append(run_batch("E1", "PE100", seeds, n, extra={"cenario": "com interferencia"}))
    batches.append(run_batch("E1", "AbI", seeds, 100, extra={"cenario": "com interferencia"}))
    runs, series = collect(batches)
    runs.to_csv(out / "e1_runs.csv", index=False)
    series.to_csv(out / "e1_series.csv", index=False)

    pe100 = runs[runs.method == "PE100"]
    rows = []
    for n, (q_ref, e_ref) in TABLE_A2.items():
        row = {"N": n, "QoF A.2": f"{q_ref:.1f}"}
        for cen, tag in [("sem interferencia", "s/ interf."), ("com interferencia", "c/ interf.")]:
            g = pe100[(pe100.N == n) & (pe100.cenario == cen)]
            row[f"QoF {tag}"] = fmt_ms(g.qof_mean.mean(), g.qof_mean.std(ddof=1), 2)
        row["Ef A.2 (%)"] = f"{100 * e_ref:.1f}"
        for cen, tag in [("sem interferencia", "s/ interf."), ("com interferencia", "c/ interf.")]:
            g = pe100[(pe100.N == n) & (pe100.cenario == cen)]
            row[f"Ef {tag} (%)"] = fmt_ms(100 * g.ef_mean.mean(), 100 * g.ef_mean.std(ddof=1), 2)
        rows.append(row)
    save_table(pd.DataFrame(rows), "e1_validacao", out)

    s100 = series[(series.N == 100) & (series.cenario == "com interferencia")]
    save_table(block_table(s100, ["method"]), "e1_abi_trechos", out)
    plots.plot_validation_abi(s100, fig / "validacao_abi.png")
    return runs


E2_TABLE_METRICS = ["ef_mean", "qof_mean", "qof_min", "viol_rate", "pe_std", "gap_oracle"]
E2_EXTRA_METRICS = ["pe_mean", "sent_total", "pe_reversals", "repairs"]


def exp_e2(seeds, out, fig):
    runs, series = collect([run_batch("E2", m, seeds, 100) for m in METHODS])
    runs.to_csv(out / "e2_runs.csv", index=False)
    series.to_csv(out / "e2_series.csv", index=False)
    summ = summarize(runs, ["method"])
    summ.to_csv(out / "e2_resumo_numerico.csv", index=False)
    table = mean_sd_table(summ, ["method"], E2_TABLE_METRICS + E2_EXTRA_METRICS)
    table = table.rename(columns={"method": "Método"})
    save_table(table, "e2_ablacao", out)

    base = runs[runs.method == "AbII"].sort_values("seed")
    rows = []
    for v in VARIANTS:
        var = runs[runs.method == v].sort_values("seed")
        row = {"Comparação": f"AbII vs {v}"}
        for m in E2_TABLE_METRICS + ["pe_mean", "sent_total"]:
            row[LABELS[m]] = wilcoxon_p(base[m].values, var[m].values)
        rows.append(row)
    wil = pd.DataFrame(rows)
    wil.to_csv(out / "e2_wilcoxon.csv", index=False)
    write_tex(wil.assign(**{c: wil[c].map(_fmt_p) for c in wil.columns[1:]}),
              out / "e2_wilcoxon.tex")
    save_table(block_table(series, ["method"]), "e2_trechos", out)

    qmin, o = 12.0, oracle(100, 12.0)
    plots.plot_time(series, "qof", qmin, fig / "qof_tempo.png", ref=qmin, ref_label="QoF$_{min}$")
    plots.plot_time(series, "pe", qmin, fig / "pe_tempo.png", ref=o.pe_star, ref_label="PE* (oráculo)")
    plots.plot_violation_box(runs, list(METHODS), fig / "ablacao_violacoes.png")
    plots.plot_ef_qof_bars(runs, list(METHODS), qmin, o.ef_star, fig / "ablacao_ef_qof.png")
    return runs


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _oracle_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    res = [oracle(int(n), q) for n, q in zip(df.N, df.qof_thr)]
    df["PE*"] = [f"{r.pe_star:.3f}" for r in res]
    df["Ef*"] = [f"{r.ef_star:.3f}" for r in res]
    df["Viável"] = ["sim" if r.feasible else "não" for r in res]
    return df


def exp_e3(seeds, out, fig):
    batches = [run_batch("E3", m, seeds, n, q) for n in (50, 100)
               for q in (9.0, 11.0, 12.0, 13.0, 15.0) for m in ("AbII", "AG-RQ")]
    runs, series = collect(batches)
    runs.to_csv(out / "e3_runs.csv", index=False)
    series.to_csv(out / "e3_series.csv", index=False)
    summ = summarize(runs, ["N", "qof_thr", "method"])
    summ.to_csv(out / "e3_resumo_numerico.csv", index=False)
    table = mean_sd_table(summ, ["N", "qof_thr", "method"],
                          ["ef_mean", "qof_mean", "qof_min", "viol_rate", "pe_mean", "gap_oracle"])
    table = _oracle_cols(table).rename(columns={"method": "Método", "qof_thr": "QoF_min"})
    save_table(table, "e3_limiar", out)
    plots.plot_threshold(summ, fig / "limiar.png")
    return runs


def exp_e4(seeds, out, fig):
    batches = [run_batch("E4", m, seeds, n, 12.0) for n in (25, 50, 100, 200)
               for m in ("AbII", "AG-RQ")]
    runs, series = collect(batches)
    runs.to_csv(out / "e4_runs.csv", index=False)
    series.to_csv(out / "e4_series.csv", index=False)
    summ = summarize(runs, ["N", "method"])
    summ["qof_thr"] = 12.0
    summ.to_csv(out / "e4_resumo_numerico.csv", index=False)
    table = mean_sd_table(summ, ["N", "qof_thr", "method"],
                          ["ef_mean", "qof_mean", "qof_min", "viol_rate", "pe_mean", "gap_oracle"])
    table = _oracle_cols(table).drop(columns="qof_thr").rename(columns={"method": "Método"})
    save_table(table, "e4_densidade", out)
    plots.plot_density(summ, 12.0, fig / "densidade.png")
    return runs


E5_SEGMENTS = {"antes (10-49)": (10, 50), "depois (50-99)": (50, 100),
               "transicao (50-59)": (50, 60)}


def exp_e5(seeds, out, fig):
    sched = dynamic_schedule(SimConfig())
    runs, series = collect([run_batch("E5", m, seeds, sched, 12.0, segments=E5_SEGMENTS)
                            for m in ("AbII", "AG-RQ")])
    runs.to_csv(out / "e5_runs.csv", index=False)
    series.to_csv(out / "e5_series.csv", index=False)
    summ = summarize(runs, ["segment", "method"])
    summ.to_csv(out / "e5_resumo_numerico.csv", index=False)
    table = mean_sd_table(summ, ["segment", "method"],
                          ["ef_mean", "qof_mean", "qof_min", "viol_rate", "pe_mean", "repairs"])
    save_table(table.rename(columns={"segment": "Trecho", "method": "Método"}), "e5_dinamico", out)
    pe_star = np.array([oracle(int(n), 12.0).pe_star for n in sched])
    plots.plot_dynamic(series, 12.0, pe_star, 50, fig / "dinamico.png")
    return runs


E6_MARGINS = (0.0, 0.05, 0.10)
E6_STEPS = (0.03, 0.06)
E6_METRICS = ["ef_mean", "qof_mean", "qof_min", "viol_rate", "pe_mean", "sent_total",
              "gap_oracle", "repairs"]


def exp_e6(seeds, out, fig):
    """Sensibilidade do reparo, definida a priori (não altera os parâmetros oficiais)."""
    batches = []
    for m in E6_MARGINS:
        for s in E6_STEPS:
            c1 = C1Config(margin=m, repair_step=s)
            batches.append(run_batch("E6", "AG-RQ", seeds, 100, 12.0, c1=c1,
                                     extra={"margin": m, "repair_step": s,
                                            "config": f"AG-RQ m={m:.2f} d={s:.2f}"}))
    for ref in ("PE100", "AbII", "AbII+Pen"):
        batches.append(run_batch("E6", ref, seeds, 100, 12.0,
                                 extra={"margin": np.nan, "repair_step": np.nan, "config": ref}))
    runs, series = collect(batches)
    runs.to_csv(out / "e6_runs.csv", index=False)
    summ = summarize(runs, ["config"])
    meta = runs.drop_duplicates("config")[["config", "method", "margin", "repair_step"]]
    summ = meta.merge(summ, on="config")
    summ["oficial"] = (summ.method == "AG-RQ") & np.isclose(summ.margin, 0.10) & \
        np.isclose(summ.repair_step, 0.06)
    summ.to_csv(out / "e6_resumo_numerico.csv", index=False)
    table = mean_sd_table(summ, ["method", "margin", "repair_step"], E6_METRICS)
    table["margin"] = [("-" if np.isnan(v) else f"{v:.2f}") for v in table["margin"]]
    table["repair_step"] = [("-" if np.isnan(v) else f"{v:.2f}") for v in table["repair_step"]]
    table = table.rename(columns={"method": "Método", "margin": "Margem m",
                                  "repair_step": "Passo δr"})
    save_table(table, "e6_sensibilidade", out)
    plots.plot_tradeoff(summ, fig / "sensibilidade_tradeoff.png")
    return runs


EXPERIMENTS = {"E1": exp_e1, "E2": exp_e2, "E3": exp_e3, "E4": exp_e4, "E5": exp_e5, "E6": exp_e6}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimentos do AG-RQ")
    parser.add_argument("--exp", default="all", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--fig", type=Path, default=FIG_DIR)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.fig.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seeds)
    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    for name in names:
        t0 = time.perf_counter()
        EXPERIMENTS[name](seeds, args.out, args.fig)
        print(f"{name}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
