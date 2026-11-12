"""Fase 4: oráculo analítico (CLAUDE.md, seção 6)."""
import numpy as np
import pytest

from src.channel import expected_received
from src.config import ChannelConfig
from src.oracle import gap, oracle

CH = ChannelConfig()


def test_reference_point():
    o = oracle(100, 12.0)
    assert o.feasible
    assert o.pe_star == pytest.approx(0.248, abs=0.001)
    assert o.ef_star == pytest.approx(0.484, abs=0.001)


@pytest.mark.parametrize("n, qmin", [(50, 9), (50, 15), (100, 13), (200, 12), (100, 15)])
def test_pe_star_hits_threshold_exactly(n, qmin):
    o = oracle(n, qmin)
    assert o.feasible and o.pe_star < 1
    k = n * o.pe_star
    assert expected_received(k, CH.r_max, CH.k_half) == pytest.approx(qmin)
    assert o.ef_star == pytest.approx(CH.r_max / (CH.k_half + k))


def test_ef_star_independent_of_n_when_feasible():
    assert oracle(100, 12).ef_star == pytest.approx(oracle(200, 12).ef_star)


def test_infeasible_case():
    o = oracle(25, 13.0)                         # QoF máx. com N = 25 é ~12,05
    assert not o.feasible and o.pe_star == 1.0
    assert o.ef_star == pytest.approx(CH.r_max / (CH.k_half + 25))


def test_feasible_but_clipped():
    o = oracle(25, 12.0)                         # viável por pouco: PE* ~ 0,99
    assert o.feasible and 0.95 < o.pe_star <= 1.0


def test_gap():
    assert gap(0.242, 0.484) == pytest.approx(0.5)
    assert gap(0.484, 0.484) == 0.0
