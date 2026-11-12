"""Métricas por execução (CLAUDE.md, seção 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ChannelConfig
from .oracle import gap, oracle

_TOL = 1e-12

METRIC_COLS = ["ef_mean", "qof_mean", "qof_min", "viol_rate", "pe_mean", "pe_std",
               "pe_reversals", "sent_total", "repairs", "repair_active", "gap_oracle"]


def count_reversals(dpe) -> int:
    """Nº de trocas de sinal entre variações de PE consecutivas não nulas."""
    d = np.asarray(dpe, dtype=float)
    signs = np.sign(d[np.abs(d) > _TOL])
    return int(np.sum(signs[1:] != signs[:-1]))


def oracle_ef_star(n_nodes, qof_min: float, ch: ChannelConfig | None = None) -> float:
    """Ef* médio das sessões (em E5 o N muda; com QoF_MIN viável, Ef* não depende de N)."""
    values = [oracle(int(n), qof_min, ch).ef_star for n in np.asarray(n_nodes)]
    return float(np.mean(values))


def compute_metrics(df: pd.DataFrame, qof_min: float, warmup: int = 10,
                    end: int | None = None, ch: ChannelConfig | None = None) -> dict:
    """Métricas de uma execução, nas sessões [warmup, end).

    - ef_mean: recebidas totais / enviadas totais;
    - viol_rate: % de sessões com QoF_t < QoF_MIN;
    - pe_std: desvio padrão (populacional) da PE usada nas sessões;
    - pe_reversals: trocas de sinal da variação efetiva da PE (PE_{t+1} − PE_t);
    - gap_oracle: (Ef* − ef_mean) / Ef*.
    """
    w = df.iloc[warmup:end]
    sent = int(w["sent"].sum())
    ef = w["received"].sum() / sent if sent > 0 else 0.0
    ef_star = oracle_ef_star(w["n_nodes"], qof_min, ch)
    return {
        "ef_mean": float(ef),
        "qof_mean": float(w["qof"].mean()),
        "qof_min": float(w["qof"].min()),
        "viol_rate": float(100.0 * (w["qof"] < qof_min).mean()),
        "pe_mean": float(w["pe"].mean()),
        "pe_std": float(w["pe"].std(ddof=0)),
        "pe_reversals": count_reversals(w["pe_next"].to_numpy() - w["pe"].to_numpy()),
        "sent_total": sent,
        "repairs": int(w["repaired"].sum()),
        "repair_active": int(w["repair_active"].sum()),
        "gap_oracle": float(gap(ef, ef_star)),
    }
