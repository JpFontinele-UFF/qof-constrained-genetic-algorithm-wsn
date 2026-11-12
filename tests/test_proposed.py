"""Fase 4: Proposed (C1 = penalidade + reparo) (CLAUDE.md, seções 5 e 9)."""
import numpy as np
import pytest

from src.config import SimConfig
from src.controllers import AbII, Proposed
from src.simulator import run

SIM = SimConfig()
QMIN = SIM.qof_min


def fresh(pen=True, rep=True, seed=0):
    c = Proposed(pen, rep)
    c.reset(np.random.default_rng(seed), SIM)
    return c


def test_names():
    assert Proposed(True, True).name == "AG-RQ"
    assert Proposed(True, False).name == "AbII+Pen"
    assert Proposed(False, True).name == "AbII+Rep"


# ------------------------------------------------------------------ penalidade
def test_fitness_equals_ef_above_threshold():
    c = fresh()
    for ef in (0.0, 0.2, 0.9):
        for qof in (QMIN, QMIN + 0.01, 25.0):
            assert c.fitness(ef, qof) == ef


def test_fitness_penalty_value():
    c = fresh()
    # QoF = 9 com mínimo 12: faltam 25% -> desconto 2 * 0,25 = 0,5
    assert c.fitness(0.4, 9.0) == pytest.approx(-0.1)


def test_penalized_criterion_equals_ef_when_both_above():
    c = fresh(pen=True, rep=False)
    c.remember(0.30, 13.0)
    assert c.is_success(0.31, 14.0) and not c.is_success(0.29, 20.0)


def test_penalized_criterion_rewards_quality_recovery():
    c = fresh(pen=True, rep=False)
    c.remember(0.50, 6.0)                       # F = 0,5 − 2·0,5 = −0,5
    assert c.is_success(0.30, 12.0)             # F = 0,3: recuperou a QoF, Ef caiu
    c.remember(0.30, 12.0)
    assert not c.is_success(0.60, 9.0)          # Ef subiu, QoF caiu: F = 0,1 < 0,3


def test_without_penalty_uses_abii_criterion():
    c = fresh(pen=False, rep=True)
    ab = AbII(); ab.reset(np.random.default_rng(0), SIM)
    for prev, cur in [((0.3, 10.0), (0.1, 11.0)), ((0.5, 6.0), (0.3, 12.0)),
                      ((0.3, 13.0), (0.29, 20.0)), ((0.2, 13.0), (0.6, 9.0))]:
        c.remember(*prev); ab.remember(*prev)
        assert c.is_success(*cur) == ab.is_success(*cur)


# --------------------------------------------------------------------- reparo
@pytest.mark.parametrize("dpe", [-0.21, -0.09, -0.03, 0.0, 0.03, 0.06, 0.09, 0.21])
def test_repair_below_threshold_forces_increase(dpe):
    c = fresh(pen=False, rep=True)
    for qof in (0.0, 5.0, QMIN - 1e-6):
        applied, active = c.adjust_action(dpe, qof)
        assert active and applied >= 0.06 and applied == max(dpe, 0.06)


@pytest.mark.parametrize("dpe", [-0.21, -0.03, 0.0, 0.03, 0.15])
def test_repair_in_margin_never_reduces(dpe):
    c = fresh(pen=False, rep=True)
    for qof in (QMIN, 12.5, QMIN * 1.1 - 1e-6):
        applied, active = c.adjust_action(dpe, qof)
        assert active and applied == max(dpe, 0.0)


@pytest.mark.parametrize("dpe", [-0.21, 0.0, 0.21])
def test_repair_inactive_with_slack(dpe):
    c = fresh(pen=False, rep=True)
    assert c.adjust_action(dpe, QMIN * 1.1) == (dpe, False)
    assert c.adjust_action(dpe, 20.0) == (dpe, False)


def test_no_repair_flag_keeps_action():
    c = fresh(pen=True, rep=False)
    assert c.adjust_action(-0.21, 0.0) == (-0.21, False)


def test_repair_never_reduces_pe_in_simulation():
    """Em execuções reais: se QoF_t < QoF_MIN, PE_{t+1} >= PE_t (+0,06 salvo o teto 1)."""
    for seed in range(10):
        for c in (Proposed(False, True), Proposed(True, True)):
            df = run(c, seed=seed)
            low = df[df.qof < QMIN]
            assert (low.pe_next >= low.pe).all()
            assert np.allclose(low.pe_next, np.minimum(1.0, low.pe + low.dpe_applied))
            assert (low.dpe_applied >= 0.06 - 1e-12).all()
            margin = df[(df.qof >= QMIN) & (df.qof < 1.1 * QMIN)]
            assert (margin.pe_next >= margin.pe).all()


def test_repair_does_not_change_stored_chromosome():
    c = fresh(pen=False, rep=True)
    c.cs.actions[0] = [0b0111] * 4              # todas: reduzir 0,21
    before = c.cs.actions.copy()
    c.update(0.2, 5.0)                          # QoF baixa, t = 0 -> ilha 0, sem evolução
    assert c.last_info["dpe_rule"] == pytest.approx(-0.21)
    assert c.last_info["dpe_applied"] == pytest.approx(0.06)
    assert c.last_info["repaired"] and c.repairs == 1
    assert np.array_equal(c.cs.actions, before)


def test_repair_counter_counts_only_changed_actions():
    c = fresh(pen=False, rep=True)
    df = run(c, seed=3)
    assert c.repairs == int(df.repaired.sum())
    assert c.repair_active == int(df.repair_active.sum()) >= c.repairs
    assert (df.repaired == (df.dpe_applied != df.dpe_rule)).all()


# ------------------------------------------------------------------- ablação
def test_proposed_without_flags_equals_abii():
    for seed in range(5):
        a = run(AbII(), seed=seed)
        b = run(Proposed(False, False), seed=seed)
        assert a.equals(b)


def test_reproducible():
    assert run(Proposed(), seed=8).equals(run(Proposed(), seed=8))
