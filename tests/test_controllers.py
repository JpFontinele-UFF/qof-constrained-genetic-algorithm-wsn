"""Fase 3: AbI e AbII (CLAUDE.md, seções 4 e 9)."""
from dataclasses import replace

import numpy as np
import pytest

from src.config import SimConfig
from src.controllers import PE100, AbI, AbII
from src.simulator import run

SIM = SimConfig()
SEEDS = range(30)
WARMUP = SIM.warmup


def fresh(cls, seed=0, sim=SIM):
    c = cls()
    c.reset(np.random.default_rng(seed), sim)
    return c


# ------------------------------------------------------ critérios de sucesso
def test_abi_success_criterion():
    c = fresh(AbI)
    c.prev_ef, c.prev_qof = 0.30, 15.0
    assert c.is_success(0.31, 1.0)       # só a Ef importa
    assert not c.is_success(0.30, 20.0)  # igual não é sucesso
    assert not c.is_success(0.29, 20.0)


def test_abii_success_above_threshold_uses_ef():
    c = fresh(AbII)
    c.prev_ef, c.prev_qof = 0.30, 5.0
    assert c.is_success(0.31, 12.0)      # QoF_t = QoF_MIN conta como acima
    assert not c.is_success(0.29, 18.0)  # QoF subiu, mas Ef caiu


def test_abii_success_below_threshold_uses_qof():
    c = fresh(AbII)
    c.prev_ef, c.prev_qof = 0.30, 10.0
    assert c.is_success(0.10, 11.0)      # QoF subiu, Ef caiu
    assert not c.is_success(0.50, 9.0)   # Ef subiu, QoF caiu
    assert not c.is_success(0.50, 10.0)


# ------------------------------------------------------------- ciclo do AG
def test_first_checkpoint_no_reward_and_island_zero():
    c = fresh(AbI)
    c.update(0.2, 15.0)
    info = c.last_info
    assert not info["rewarded"] and info["island"] == 0  # ΔEf = 0 -> "caiu/igual", nível 0
    assert c.cs.bank == pytest.approx(100.0)
    assert c.cs.total_wealth() == pytest.approx(64 * 1000.0)


def test_rule_paid_then_rewarded():
    c = fresh(AbI)
    c.update(0.2, 15.0)
    island, slot = c.prev_rule
    budget_after_fee = c.cs.budgets[island, slot]
    c.update(0.3, 15.0)                  # Ef subiu -> regra anterior recebe o banco (100)
    assert c.last_info["rewarded"]
    assert budget_after_fee == pytest.approx(900.0)
    assert c.cs.budgets[island, slot] == pytest.approx(1000.0)  # recebeu o banco (100)
    assert c.last_info["island"] == 12   # ΔEf = +0,1 -> sinal 1, nível 4: outra ilha
    assert c.cs.bank == pytest.approx(100.0)  # só a cobrança da nova regra
    assert c.n_rewards == 1


def test_evolution_every_two_checkpoints():
    c = fresh(AbII)
    evolved = []
    for t in range(10):
        c.update(0.2 + 0.01 * t, 15.0)
        evolved.append(c.last_info["evolved"])
    assert evolved == [False, True] * 5
    assert c.cs.n_evolutions == 5


def test_applied_action_equals_rule_action():
    c = fresh(AbII)
    for t in range(50):
        pe_before = c.pe
        c.update(0.2, 8.0 if t % 2 else 14.0)
        info = c.last_info
        assert info["dpe_applied"] == info["dpe_rule"] and not info["repaired"]
        assert c.pe == pytest.approx(min(1.0, max(0.01, pe_before + info["dpe_rule"])))


def test_population_and_bounds_during_runs():
    for cls in (AbI, AbII):
        c = cls()
        df = run(c, seed=1)
        assert df["pe"].between(SIM.pe_min, 1.0).all()
        assert c.cs.size == 64
        assert c.cs.budgets.min() >= 1.0


def test_reproducible():
    assert run(AbII(), seed=2).equals(run(AbII(), seed=2))


def test_abii_equals_abi_when_threshold_never_binds():
    """Com QoF_MIN = 0 o critério da AbII se reduz ao da AbI."""
    sim0 = replace(SIM, qof_min=0.0)
    assert run(AbI(), seed=4, sim_cfg=sim0).equals(run(AbII(), seed=4, sim_cfg=sim0))


# ----------------------------------------------- critério qualitativo (fase 3)
def _summary(cls):
    dfs = [run(cls(), seed=s, n_nodes=100).iloc[WARMUP:] for s in SEEDS]
    pe = np.array([d["pe"].values for d in dfs])
    qof = np.array([d["qof"].values for d in dfs])
    return pe, qof


@pytest.fixture(scope="module")
def summaries():
    return {cls.name: _summary(cls) for cls in (PE100, AbI, AbII)}


def test_abi_lowers_pe_and_qof(summaries):
    """AbI leva a PE a valores baixos e o QoF cai (abaixo da rede sem controle)."""
    pe, qof = summaries["AbI"]
    _, qof100 = summaries["PE100"]
    assert pe.mean() < 0.6
    assert pe[:, -30:].mean() < pe[:, :20].mean()     # a PE cai ao longo da execução
    assert qof[:, -30:].mean() < qof[:, :20].mean()   # e o QoF também
    assert qof.mean() < qof100.mean() - 3.0


def test_abii_keeps_qof_near_threshold_with_oscillation(summaries):
    """AbII mantém o QoF médio perto do limiar (com oscilação da PE)."""
    pe, qof = summaries["AbII"]
    pe_abi, qof_abi = summaries["AbI"]
    assert abs(qof.mean() - SIM.qof_min) <= 1.5
    assert pe.std(axis=1).mean() > 0.1
    assert qof[:, -30:].mean() > qof_abi[:, -30:].mean()
