"""Fase 1: canal, simulador e PE100 (CLAUDE.md, seções 3 e 9)."""
from dataclasses import replace

import numpy as np
import pytest

from src.channel import draw_interference, expected_received, simulate_session, success_prob
from src.config import ChannelConfig, SimConfig
from src.controllers import PE100, FixedPE
from src.simulator import dynamic_schedule, run

CH = ChannelConfig()
SIM_NO_INTERF = SimConfig(channel=replace(CH, p_interf=0.0))
SEEDS = range(30)

# Tabela A.2 da dissertação (rede sem controle, PE = 100%)
TABLE_A2 = {25: (12.0, 0.49), 50: (16.0, 0.322), 100: (19.0, 0.19), 200: (21.0, 0.105)}


def test_success_prob_in_unit_interval():
    k = np.arange(0, 1001)
    for r in (CH.r_max, CH.r_interf):
        s = success_prob(k, r, CH.k_half)
        assert np.all(s > 0) and np.all(s <= 1)
        assert np.all(np.diff(s) < 0)  # cai com k


def test_expected_received_saturates():
    k = np.array([1, 10, 100, 1000, 10**6])
    e = expected_received(k, CH.r_max, CH.k_half)
    assert np.all(np.diff(e) > 0) and e[-1] < CH.r_max
    assert e[-1] == pytest.approx(CH.r_max, rel=1e-3)


def test_session_mean_matches_model():
    rng = np.random.default_rng(123)
    n_rounds = 200_000
    sent, rec = simulate_session(rng, 100, 1.0, CH.r_max, CH.k_half, n_rounds)
    assert sent == 100 * n_rounds
    assert rec / n_rounds == pytest.approx(expected_received(100, CH.r_max, CH.k_half), abs=0.02)


def test_interference_frequency_and_reproducibility():
    a = draw_interference(np.random.default_rng(7), 100_000, CH.p_interf)
    assert a.mean() == pytest.approx(0.10, abs=0.005)
    b1 = draw_interference(np.random.default_rng(7), 100, CH.p_interf)
    b2 = draw_interference(np.random.default_rng(7), 100, CH.p_interf)
    assert np.array_equal(b1, b2)


def test_same_interference_for_all_methods():
    df1 = run(PE100(), seed=5)
    df2 = run(FixedPE(0.3), seed=5)
    assert df1["interf"].equals(df2["interf"])
    assert df1["interf"].any()


def test_run_shape_and_bounds():
    df = run(FixedPE(0.5), seed=0)
    sim = SimConfig()
    assert len(df) == sim.n_sessions == 100
    assert (df["sent"] >= df["received"]).all()
    assert df["qof"].equals(df["received"] / 19)
    assert df["pe"].between(sim.pe_min, 1.0).all()


def test_reproducible_by_seed():
    assert run(PE100(), seed=3).equals(run(PE100(), seed=3))
    assert not run(PE100(), seed=3).equals(run(PE100(), seed=4))


def test_dynamic_schedule():
    sched = dynamic_schedule(SimConfig())
    assert (sched[:50] == 100).all() and (sched[50:] == 200).all()
    df = run(PE100(), seed=0, n_nodes=sched, sim_cfg=SIM_NO_INTERF)
    assert (df.loc[df.t >= 50, "sent"] == 200 * 19).all()


@pytest.mark.parametrize("n_nodes", sorted(TABLE_A2))
def test_pe100_reproduces_table_a2(n_nodes):
    """Critério de aceite da Fase 1: QoF a ±0,5 e Ef a ±2 p.p. da Tabela A.2."""
    qof_target, ef_target = TABLE_A2[n_nodes]
    dfs = [run(PE100(), seed=s, n_nodes=n_nodes, sim_cfg=SIM_NO_INTERF) for s in SEEDS]
    qof = np.mean([d["qof"].mean() for d in dfs])
    ef = sum(d["received"].sum() for d in dfs) / sum(d["sent"].sum() for d in dfs)
    assert abs(qof - qof_target) <= 0.5
    assert abs(ef - ef_target) <= 0.02
