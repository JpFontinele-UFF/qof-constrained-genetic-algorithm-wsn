"""Fase 2: sistema classificador e AG (CLAUDE.md, seções 4 e 9)."""
import numpy as np
import pytest

from src.classifier import (
    ClassifierSystem, action_to_dpe, bits_to_half, chromosome, condition_from_delta,
    decode_half, encode_half, half_to_bits, split_chromosome,
)
from src.config import GAConfig

CFG = GAConfig()


# ------------------------------------------------------------- decodificação
def test_encode_decode_roundtrip():
    for sign in (0, 1):
        for level in range(8):
            code = encode_half(sign, level)
            assert decode_half(code) == (sign, level)
            assert bits_to_half(half_to_bits(code)) == code
            assert half_to_bits(code)[0] == sign  # bit 0 = sinal


def test_bit_order_example():
    # sinal 1, nível 5 (101b) -> bits [1, 1, 0, 1]
    assert half_to_bits(encode_half(1, 5)) == [1, 1, 0, 1]
    assert half_to_bits(encode_half(0, 1)) == [0, 0, 0, 1]


def test_chromosome_split():
    for cond in range(16):
        for act in range(16):
            assert split_chromosome(chromosome(cond, act)) == (cond, act)
    assert chromosome(0b1010, 0b0011) == 0b10100011


@pytest.mark.parametrize("delta, sign, level", [
    (0.0, 0, 0),            # igual -> sinal 0
    (0.01, 1, 0),
    (-0.01, 0, 0),
    (0.025, 1, 1),          # limite exato da faixa
    (-0.05, 0, 2),
    (0.075, 1, 3),          # 0.075/0.025 = 2.9999… em ponto flutuante
    (0.174, 1, 6),
    (0.175, 1, 7),
    (-0.9, 0, 7),           # satura em 7
])
def test_condition_from_delta(delta, sign, level):
    assert decode_half(condition_from_delta(delta)) == (sign, level)


@pytest.mark.parametrize("sign, level, dpe", [
    (1, 0, 0.0), (0, 0, 0.0), (1, 1, 0.03), (0, 1, -0.03), (1, 7, 0.21), (0, 7, -0.21),
])
def test_action_to_dpe(sign, level, dpe):
    assert action_to_dpe(encode_half(sign, level)) == pytest.approx(dpe)


# ---------------------------------------------------------------- população
def test_initial_population():
    cs = ClassifierSystem(np.random.default_rng(0))
    assert cs.actions.shape == (16, 4) and cs.size == 64
    assert np.all(cs.budgets == 1000.0) and cs.bank == 0.0
    assert cs.actions.min() >= 0 and cs.actions.max() <= 15


def test_initial_actions_uniform():
    acts = np.concatenate([ClassifierSystem(np.random.default_rng(s)).actions.ravel()
                           for s in range(500)])
    freq = np.bincount(acts, minlength=16) / acts.size
    assert np.allclose(freq, 1 / 16, atol=0.01)


# ------------------------------------------------------------------- banco
def test_charge_and_reward():
    cs = ClassifierSystem(np.random.default_rng(0))
    assert cs.charge(3, 2) == 100.0
    assert cs.budgets[3, 2] == 900.0 and cs.bank == 100.0
    cs.charge(5, 0)
    assert cs.bank == 200.0
    assert cs.reward(5, 0) == 200.0
    assert cs.budgets[5, 0] == 1100.0 and cs.bank == 0.0


def test_budget_floor_without_creating_money():
    cs = ClassifierSystem(np.random.default_rng(0))
    for _ in range(15):
        cs.charge(0, 0)
    assert cs.budgets[0, 0] == pytest.approx(1.0)
    assert cs.bank == pytest.approx(999.0)
    assert cs.charge(0, 0) == 0.0  # no piso não paga mais nada
    assert cs.budgets[0, 0] == 1.0


def test_wealth_conserved_without_evolution():
    rng = np.random.default_rng(42)
    cs = ClassifierSystem(rng)
    w0 = cs.total_wealth()
    last = None
    for _ in range(5000):
        if last is not None and rng.random() < 0.3:
            cs.reward(*last)
        island = int(rng.integers(16))
        slot = cs.choose(island)
        cs.charge(island, slot)
        last = (island, slot)
        assert cs.total_wealth() == pytest.approx(w0)
        assert cs.budgets.min() >= CFG.min_budget
    assert cs.budgets.min() == pytest.approx(1.0)  # o piso foi atingido no teste


def test_choose_is_uniform():
    cs = ClassifierSystem(np.random.default_rng(1))
    picks = np.array([cs.choose(0) for _ in range(40_000)])
    assert np.allclose(np.bincount(picks, minlength=4) / picks.size, 0.25, atol=0.01)


# ----------------------------------------------------------------- evolução
def test_evolution_never_changes_condition_and_keeps_size():
    cs = ClassifierSystem(np.random.default_rng(7))
    for _ in range(3000):
        island = int(cs.rng.integers(16))
        slot = cs.choose(island)
        cs.charge(island, slot)
        if cs.rng.random() < 0.4:
            cs.reward(island, slot)
        cs.evolve()
        chrom = cs.chromosomes()
        assert chrom.shape == (16, 4) and cs.size == 64
        conds = chrom >> 4
        assert np.all(conds == np.arange(16)[:, None])
        assert cs.budgets.min() >= CFG.min_budget


def test_evolution_only_touches_one_slot_of_one_island():
    cs = ClassifierSystem(np.random.default_rng(3))
    before_a, before_b = cs.actions.copy(), cs.budgets.copy()
    rec = cs.evolve(island=5)
    diff = (cs.actions != before_a) | (cs.budgets != before_b)
    mask = np.zeros_like(diff)
    mask[5, rec["target"]] = True
    assert not np.any(diff & ~mask)


def test_parents_are_top_two_and_child_budget_is_mean():
    cs = ClassifierSystem(np.random.default_rng(11))
    cs.budgets[2] = [50.0, 3000.0, 10.0, 1500.0]
    cs.actions[2] = [0b0000, 0b1111, 0b0101, 0b0000]
    rec = cs.evolve(island=2)
    assert set(rec["parents"]) == {1, 3}
    assert rec["child_budget"] == pytest.approx(2250.0)
    assert cs.budgets[2, rec["target"]] == pytest.approx(2250.0)


def test_crossover_one_point_without_mutation():
    cfg = GAConfig(mutation_rate=0.0)
    for seed in range(200):
        cs = ClassifierSystem(np.random.default_rng(seed), cfg)
        cs.budgets[0] = [10.0, 900.0, 800.0, 5.0]
        cs.actions[0] = [0, 0b1111, 0b0000, 0]
        rec = cs.evolve(island=0)
        bits = half_to_bits(rec["child"])
        p1, _ = rec["parents"]
        first = 1 if p1 == 1 else 0  # pai 1 = 1111, pai 2 = 0000
        cut = rec["cut"]
        assert cut in (1, 2, 3)
        assert bits == [first] * cut + [1 - first] * (4 - cut)


def test_mutation_rate():
    cfg = GAConfig()
    cs = ClassifierSystem(np.random.default_rng(5), cfg)
    n = 50_000
    rate = sum(cs.evolve()["mutated"] for _ in range(n)) / n
    assert rate == pytest.approx(0.01, abs=0.002)


def test_mutation_flips_exactly_one_bit():
    cfg = GAConfig(mutation_rate=1.0)
    for seed in range(100):
        cs = ClassifierSystem(np.random.default_rng(seed), cfg)
        cs.actions[0] = [0b1010] * 4  # pais iguais -> filho sem mutação seria 1010
        rec = cs.evolve(island=0)
        assert bin(rec["child"] ^ 0b1010).count("1") == 1


def test_reproducible():
    a = ClassifierSystem(np.random.default_rng(9))
    b = ClassifierSystem(np.random.default_rng(9))
    for _ in range(100):
        assert a.evolve() == b.evolve()
    assert np.array_equal(a.actions, b.actions)
