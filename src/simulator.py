"""Laço de sessões/rounds (CLAUDE.md, seção 3)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .channel import draw_interference, effective_capacity, simulate_session
from .config import SimConfig
from .controllers import Controller

CTRL_SEED_OFFSET = 10_000


def make_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Dois geradores por semente: canal e controlador (números aleatórios comuns)."""
    return np.random.default_rng(seed), np.random.default_rng(seed + CTRL_SEED_OFFSET)


def n_schedule(sim_cfg: SimConfig, n_nodes: int | Sequence[int]) -> np.ndarray:
    """Número de sensores em cada sessão (constante ou lista por sessão)."""
    if np.isscalar(n_nodes):
        return np.full(sim_cfg.n_sessions, int(n_nodes))
    sched = np.asarray(n_nodes, dtype=int)
    if sched.shape != (sim_cfg.n_sessions,):
        raise ValueError("n_nodes deve ter um valor por sessão")
    return sched


def dynamic_schedule(sim_cfg: SimConfig, n_before: int = 100, n_after: int = 200,
                     switch_session: int = 50) -> np.ndarray:
    """Cenário E5: N passa de n_before para n_after no início de switch_session."""
    sched = np.full(sim_cfg.n_sessions, n_before)
    sched[switch_session:] = n_after
    return sched


def run(controller: Controller, seed: int, n_nodes: int | Sequence[int] = 100,
        sim_cfg: SimConfig | None = None) -> pd.DataFrame:
    """Executa uma simulação completa e devolve uma linha por sessão.

    Colunas: t, n_nodes, interf, pe (usada na sessão), sent, received, ef, qof e a
    decisão tomada no checkpoint ao fim da sessão (dpe_rule, dpe_applied, repaired,
    repair_active, pe_next).
    """
    sim_cfg = sim_cfg or SimConfig()
    ch = sim_cfg.channel
    rng_channel, rng_ctrl = make_rngs(seed)
    sched = n_schedule(sim_cfg, n_nodes)
    # Interferência sorteada antes de tudo: mesma sequência para todos os métodos.
    interf = draw_interference(rng_channel, sim_cfg.n_sessions, ch.p_interf)

    pe = controller.reset(rng_ctrl, sim_cfg)
    m = sim_cfg.monitored_rounds
    rows = []
    for t in range(sim_cfg.n_sessions):
        r_eff = effective_capacity(bool(interf[t]), ch)
        sent, received = simulate_session(rng_channel, int(sched[t]), pe, r_eff, ch.k_half, m)
        ef = received / sent if sent > 0 else 0.0
        qof = received / m
        pe_next = float(min(1.0, max(sim_cfg.pe_min, controller.update(ef, qof))))
        info = controller.last_info
        rows.append({
            "t": t, "n_nodes": int(sched[t]), "interf": bool(interf[t]), "pe": pe,
            "sent": sent, "received": received, "ef": ef, "qof": qof,
            "dpe_rule": info.get("dpe_rule", 0.0),
            "dpe_applied": info.get("dpe_applied", 0.0),
            "repaired": info.get("repaired", False),
            "repair_active": info.get("repair_active", False),
            "pe_next": pe_next,
        })
        pe = pe_next
    return pd.DataFrame(rows)
