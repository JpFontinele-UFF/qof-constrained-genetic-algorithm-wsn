"""Oráculo analítico: PE ótima do modelo de canal sem interferência (CLAUDE.md, seção 6).

Com k = N·PE transmissores esperados: QoF = R_MAX·k/(K_HALF + k) e
Ef = R_MAX/(K_HALF + k). Como Ef cai e QoF sobe com a PE, o ótimo de
max Ef s.a. QoF >= QoF_MIN é a menor PE viável.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ChannelConfig


@dataclass(frozen=True)
class OracleResult:
    pe_star: float
    ef_star: float
    k_star: float
    feasible: bool


def oracle(n_nodes: int, qof_min: float, ch: ChannelConfig | None = None) -> OracleResult:
    ch = ch or ChannelConfig()
    qof_max = ch.r_max * n_nodes / (ch.k_half + n_nodes)
    if qof_min >= qof_max:
        # Inviável: nem com PE = 1 a rede alcança QoF_MIN.
        pe_star, feasible = 1.0, False
    else:
        k_star = ch.k_half * qof_min / (ch.r_max - qof_min)
        pe_star, feasible = min(1.0, k_star / n_nodes), True
    k = n_nodes * pe_star
    return OracleResult(pe_star, ch.r_max / (ch.k_half + k), k, feasible)


def gap(ef: float, ef_star: float) -> float:
    """Distância relativa ao ótimo: (Ef* − Ef) / Ef*."""
    return (ef_star - ef) / ef_star
