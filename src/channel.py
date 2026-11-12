"""Modelo probabilístico do canal compartilhado (CLAUDE.md, seção 3)."""
from __future__ import annotations

import numpy as np

from .config import ChannelConfig


def success_prob(k, r_eff: float, k_half: float):
    """Probabilidade de cada transmissão chegar no prazo: s(k) = R_eff / (K_HALF + k).

    Com os parâmetros calibrados (R_MAX < K_HALF) s(k) <= 1 para todo k >= 0;
    o np.minimum é só uma proteção caso os parâmetros sejam alterados.
    """
    return np.minimum(1.0, r_eff / (k_half + np.asarray(k, dtype=float)))


def expected_received(k, r_eff: float, k_half: float):
    """Número esperado de mensagens recebidas por round com k transmissores."""
    k = np.asarray(k, dtype=float)
    return k * success_prob(k, r_eff, k_half)


def draw_interference(rng: np.random.Generator, n_sessions: int, p_interf: float) -> np.ndarray:
    """Sorteia, antecipadamente, quais sessões sofrem interferência."""
    return rng.random(n_sessions) < p_interf


def effective_capacity(interf: bool, cfg: ChannelConfig) -> float:
    return cfg.r_interf if interf else cfg.r_max


def simulate_session(
    rng: np.random.Generator,
    n_nodes: int,
    pe: float,
    r_eff: float,
    k_half: float,
    n_rounds: int,
) -> tuple[int, int]:
    """Simula os rounds monitorados de uma sessão.

    Em cada round: k ~ Binomial(N, PE) sensores transmitem e
    r ~ Binomial(k, s(k)) mensagens chegam no prazo.
    Retorna (enviadas, recebidas) somadas na sessão.
    """
    k = rng.binomial(n_nodes, pe, size=n_rounds)
    r = rng.binomial(k, success_prob(k, r_eff, k_half))
    return int(k.sum()), int(r.sum())
