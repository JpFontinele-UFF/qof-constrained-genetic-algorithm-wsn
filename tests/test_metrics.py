"""Fase 5: métricas e experimentos (CLAUDE.md, seções 7 e 8)."""
import numpy as np
import pandas as pd
import pytest

from src import experiments
from src.metrics import compute_metrics, count_reversals
from src.oracle import oracle


def synthetic(n=20, warmup_junk=True):
    """Série sintética com valores conhecidos."""
    t = np.arange(n)
    pe = np.array([0.5, 0.6, 0.6, 0.4, 0.5, 0.5, 0.3, 0.3, 0.4, 0.2] * (n // 10))
    pe_next = np.r_[pe[1:], pe[-1]]
    qof = np.where(t % 4 == 0, 10.0, 14.0)
    sent = np.full(n, 100)
    received = np.full(n, 20)
    received[::2] = 40
    df = pd.DataFrame({"t": t, "n_nodes": 100, "interf": False, "pe": pe, "pe_next": pe_next,
                       "sent": sent, "received": received, "ef": received / sent, "qof": qof,
                       "dpe_rule": 0.0, "dpe_applied": 0.0, "repaired": t % 5 == 0,
                       "repair_active": t % 5 == 0})
    if warmup_junk:  # sessões de aquecimento com valores absurdos
        df.loc[:9, ["qof"]] = 0.0
        df.loc[:9, ["received"]] = 0
    return df


def test_count_reversals():
    assert count_reversals([0.03, 0.03, -0.03, 0, 0, -0.06, 0.09, 0]) == 2
    assert count_reversals([0, 0, 0]) == 0
    assert count_reversals([0.1, -0.1, 0.1, -0.1]) == 3


def test_metrics_values_and_warmup():
    df = synthetic(20)
    m = compute_metrics(df, qof_min=12.0, warmup=10)
    w = df.iloc[10:]
    assert m["ef_mean"] == pytest.approx(w.received.sum() / w.sent.sum())   # razão de totais
    assert m["ef_mean"] == pytest.approx(0.30)
    assert m["qof_mean"] == pytest.approx(w.qof.mean())
    assert m["qof_min"] == 10.0                                    # aquecimento (0) ignorado
    assert m["viol_rate"] == pytest.approx(100 * (w.qof < 12).mean())
    assert m["viol_rate"] == pytest.approx(20.0)                  # t = 12 e 16
    assert m["pe_mean"] == pytest.approx(w.pe.mean())
    assert m["pe_std"] == pytest.approx(w.pe.std(ddof=0))
    assert m["sent_total"] == 1000
    assert m["repairs"] == 2
    ef_star = oracle(100, 12.0).ef_star
    assert m["gap_oracle"] == pytest.approx((ef_star - 0.30) / ef_star)


def test_metrics_reversals_use_effective_pe_change():
    df = synthetic(20)
    m = compute_metrics(df, 12.0, 10)
    w = df.iloc[10:]
    assert m["pe_reversals"] == count_reversals(w.pe_next.values - w.pe.values)


def test_metrics_zero_sent():
    df = synthetic(20)
    df["sent"] = 0
    df["received"] = 0
    assert compute_metrics(df, 12.0, 10)["ef_mean"] == 0.0


def test_metrics_segment():
    df = synthetic(30, warmup_junk=False)
    m = compute_metrics(df, 12.0, warmup=10, end=20)
    assert m["sent_total"] == 1000


def test_wilcoxon_helper():
    assert experiments.wilcoxon_p([1, 2, 3], [1, 2, 3]) == 1.0
    x = np.arange(30, dtype=float)
    assert experiments.wilcoxon_p(x, x + 1 + 0.01 * x) < 0.001


def test_write_tex(tmp_path):
    df = pd.DataFrame({"Método": ["AbII+Pen", "AG-RQ"], "Violação (%)": ["33.6 (25.5)", "9.6 (7.5)"]})
    experiments.write_tex(df, tmp_path / "t.tex")
    tex = (tmp_path / "t.tex").read_text(encoding="utf-8")
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert r"Violação (\%)" in tex and "33{,}6 (25{,}5)" in tex


def test_e2_smoke(tmp_path):
    """Smoke test do E2 com 2 sementes: arquivos gerados e coerentes."""
    out, fig = tmp_path / "results", tmp_path / "figuras"
    out.mkdir(); fig.mkdir()
    runs = experiments.exp_e2(range(2), out, fig)
    assert set(runs.method) == set(experiments.METHODS)
    assert len(runs) == 2 * len(experiments.METHODS)
    for f in ("e2_runs.csv", "e2_series.csv", "e2_ablacao.csv", "e2_ablacao.tex",
              "e2_wilcoxon.csv", "e2_wilcoxon.tex", "e2_trechos.csv"):
        assert (out / f).exists(), f
    for f in ("qof_tempo.png", "pe_tempo.png", "ablacao_violacoes.png", "ablacao_ef_qof.png"):
        assert (fig / f).stat().st_size > 10_000, f
    series = pd.read_csv(out / "e2_series.csv")
    assert len(series) == 2 * len(experiments.METHODS) * 100
    assert (runs[runs.method == "PE100"].pe_mean == 1.0).all()


def test_threshold_column_not_overwritten_by_metric():
    """O limiar (qof_thr) e a métrica QoF mínima (qof_min) são colunas distintas."""
    runs, _ = experiments.run_batch("T", "AbII", range(2), 50, qof_min=9.0)
    assert (runs.qof_thr == 9.0).all()
    assert not np.allclose(runs.qof_min, 9.0)
