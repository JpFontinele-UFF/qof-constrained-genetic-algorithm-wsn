"""Parâmetros do simulador, do AG e da contribuição C1 (CLAUDE.md, seções 3–5).

Os valores padrão são os da especificação. Qualquer alteração deve ser aprovada e
registrada em results/ajustes.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelConfig:
    """Modelo de canal s(k) = R_eff / (K_HALF + k) (seção 3)."""

    r_max: float = 23.52          # capacidade do canal (msg/round), calibrada pela Tabela A.2
    k_half: float = 23.81         # constante de saturação, calibrada pela Tabela A.2
    p_interf: float = 0.10        # probabilidade de uma sessão ter interferência
    interf_drop: float = 0.25     # redução relativa de R_MAX sob interferência

    @property
    def r_interf(self) -> float:
        return self.r_max * (1.0 - self.interf_drop)


@dataclass(frozen=True)
class SimConfig:
    """Tempo, limites de PE e protocolo de medição (seções 3 e 7)."""

    total_rounds: int = 2000
    session_len: int = 20
    overhead_rounds: int = 1      # rodada ocupada pelo checkpoint
    pe_min: float = 0.01
    pe_init: float = 1.0
    qof_min: float = 12.0
    warmup: int = 10              # sessões descartadas nas métricas
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    @property
    def n_sessions(self) -> int:
        return self.total_rounds // self.session_len

    @property
    def monitored_rounds(self) -> int:
        return self.session_len - self.overhead_rounds


@dataclass(frozen=True)
class GAConfig:
    """Sistema classificador + AG original (seção 4)."""

    n_islands: int = 16
    actions_per_island: int = 4
    initial_budget: float = 1000.0
    fee_fraction: float = 0.10    # fração do budget INICIAL paga ao banco
    min_budget: float = 1.0
    evol_period: int = 2
    mutation_rate: float = 0.01
    delta_ef_step: float = 0.025  # largura das faixas de |ΔEf| (bits 1–3 da condição)
    delta_pe_step: float = 0.03   # ΔPE por nível da ação

    @property
    def fee(self) -> float:
        return self.fee_fraction * self.initial_budget


@dataclass(frozen=True)
class C1Config:
    """Contribuição C1: penalidade e reparo (seção 5)."""

    lam: float = 2.0
    margin: float = 0.10
    repair_step: float = 0.06
