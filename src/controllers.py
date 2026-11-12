"""Controladores da PE executados no nó coordenador (CLAUDE.md, seções 4–5).

Fase 1: só a interface e o PE100 (sem controle).
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig


class Controller:
    """Interface: o simulador chama reset() uma vez e update() a cada checkpoint.

    update() recebe as medidas da sessão que terminou e devolve a PE da próxima.
    Depois de cada update(), `last_info` descreve a decisão tomada (para os logs).
    """

    name = "base"

    def reset(self, rng_ctrl: np.random.Generator, sim_cfg: SimConfig) -> float:
        """Reinicia o estado e devolve a PE inicial."""
        self.rng = rng_ctrl
        self.sim_cfg = sim_cfg
        self.pe = sim_cfg.pe_init
        self.last_info: dict = {}
        return self.pe

    def update(self, ef: float, qof: float) -> float:
        raise NotImplementedError

    def clip_pe(self, pe: float) -> float:
        return float(min(1.0, max(self.sim_cfg.pe_min, pe)))


class PE100(Controller):
    """Rede sem controle: PE = 1.0 sempre."""

    name = "PE100"

    def reset(self, rng_ctrl, sim_cfg):
        super().reset(rng_ctrl, sim_cfg)
        self.pe = 1.0
        return self.pe

    def update(self, ef, qof):
        self.last_info = {"dpe_rule": 0.0, "dpe_applied": 0.0,
                          "repaired": False, "repair_active": False}
        return self.pe


class FixedPE(Controller):
    """PE constante (auxiliar para testes e validação do canal)."""

    def __init__(self, pe: float):
        self.fixed_pe = pe
        self.name = f"Fixed({pe:g})"

    def reset(self, rng_ctrl, sim_cfg):
        super().reset(rng_ctrl, sim_cfg)
        self.pe = self.clip_pe(self.fixed_pe)
        return self.pe

    def update(self, ef, qof):
        self.last_info = {"dpe_rule": 0.0, "dpe_applied": 0.0,
                          "repaired": False, "repair_active": False}
        return self.pe
