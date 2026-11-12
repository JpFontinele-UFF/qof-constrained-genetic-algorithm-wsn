"""Sistema classificador + AG original de Bitencort Jr. (CLAUDE.md, seção 4).

Cromossomo de 8 bits = 4 bits de condição + 4 bits de ação. Em cada metade, o
bit 0 é o mais à esquerda (sinal) e os bits 1–3 são o nível em binário, com o
bit 1 como mais significativo. Assim, cada metade é um inteiro 0..15:
    código = sinal * 8 + nível.
A condição de um cromossomo é o índice da sua ilha; por isso a população guarda
só as ações (16 ilhas × 4 ações) e o cruzamento nunca pode alterar a condição.
"""
from __future__ import annotations

import math

import numpy as np

from .config import GAConfig

N_BITS_HALF = 4
LEVEL_MAX = 7
_EPS = 1e-9  # tolerância contra erro de ponto flutuante na divisão |ΔEf| / passo


# ---------------------------------------------------------------- codificação
def encode_half(sign: int, level: int) -> int:
    """(sinal, nível) -> código de 4 bits (0..15)."""
    if sign not in (0, 1) or not 0 <= level <= LEVEL_MAX:
        raise ValueError("sinal deve ser 0/1 e nível 0..7")
    return sign * 8 + level


def decode_half(code: int) -> tuple[int, int]:
    """Código de 4 bits -> (sinal, nível)."""
    if not 0 <= code <= 15:
        raise ValueError("código deve estar em 0..15")
    return code >> 3, code & 0b111


def half_to_bits(code: int) -> list[int]:
    """Código -> lista [bit0 (sinal), bit1, bit2, bit3]."""
    return [(code >> (3 - i)) & 1 for i in range(N_BITS_HALF)]


def bits_to_half(bits) -> int:
    code = 0
    for b in bits:
        code = (code << 1) | int(b)
    return code


def chromosome(condition: int, action: int) -> int:
    """Cromossomo completo de 8 bits: condição nos 4 bits da esquerda."""
    return (condition << 4) | action


def split_chromosome(chrom: int) -> tuple[int, int]:
    return chrom >> 4, chrom & 0b1111


def condition_from_delta(delta_ef: float, step: float = 0.025) -> int:
    """Condição (= ilha) a partir de ΔEf = Ef_t − Ef_{t−1}.

    Sinal 1 se Ef aumentou, 0 se caiu ou ficou igual;
    nível = min(7, floor(|ΔEf| / step)).
    """
    sign = 1 if delta_ef > 0 else 0
    level = min(LEVEL_MAX, math.floor(abs(delta_ef) / step + _EPS))
    return encode_half(sign, level)


def action_to_dpe(action: int, step: float = 0.03) -> float:
    """Ação -> ΔPE = ±nível·step (sinal 1 aumenta, 0 reduz; nível 0 = não muda)."""
    sign, level = decode_half(action)
    return (1.0 if sign else -1.0) * level * step


# ------------------------------------------------------- população + banco + AG
class ClassifierSystem:
    """População de 16 ilhas × 4 regras, com budgets e um banco comum."""

    def __init__(self, rng: np.random.Generator, cfg: GAConfig | None = None):
        self.cfg = cfg or GAConfig()
        self.rng = rng
        shape = (self.cfg.n_islands, self.cfg.actions_per_island)
        self.actions = rng.integers(0, 16, size=shape)            # ações iniciais uniformes
        self.budgets = np.full(shape, self.cfg.initial_budget, dtype=float)
        self.bank = 0.0
        self.n_evolutions = 0

    # -- consultas
    @property
    def size(self) -> int:
        return self.actions.size

    def chromosomes(self) -> np.ndarray:
        """Matriz 16×4 com os cromossomos de 8 bits."""
        cond = np.arange(self.cfg.n_islands)[:, None]
        return (cond << 4) | self.actions

    def total_wealth(self) -> float:
        return float(self.budgets.sum() + self.bank)

    def dpe(self, island: int, slot: int) -> float:
        return action_to_dpe(int(self.actions[island, slot]), self.cfg.delta_pe_step)

    # -- banco
    def choose(self, island: int) -> int:
        """Escolha aleatória uniforme de uma regra da ilha (como no original)."""
        return int(self.rng.integers(self.cfg.actions_per_island))

    def charge(self, island: int, slot: int) -> float:
        """A regra escolhida paga 10% do budget inicial ao banco.

        Para respeitar o piso (budget >= 1) sem criar dinheiro, paga no máximo
        budget − 1: pagamento = min(taxa, budget − piso).
        """
        fee = min(self.cfg.fee, max(0.0, self.budgets[island, slot] - self.cfg.min_budget))
        self.budgets[island, slot] -= fee
        self.bank += fee
        return fee

    def reward(self, island: int, slot: int) -> float:
        """Regra bem-sucedida recebe todo o montante do banco; o banco zera."""
        amount = self.bank
        self.budgets[island, slot] += amount
        self.bank = 0.0
        return amount

    # -- evolução
    def _top_two(self, island: int) -> tuple[int, int]:
        """Os 2 slots de maior budget na ilha (empates desfeitos por sorteio)."""
        b = self.budgets[island]
        tiebreak = self.rng.random(b.size)
        order = np.lexsort((tiebreak, -b))
        return int(order[0]), int(order[1])

    def evolve(self, island: int | None = None) -> dict:
        """Uma evolução: 1 filho por cruzamento de 1 ponto nos 4 bits de ação.

        - ilha sorteada (se não for dada);
        - pais = 2 maiores budgets; ordem dos pais sorteada;
        - ponto de corte uniforme em {1, 2, 3}: o filho herda bits[:corte] do
          primeiro pai e bits[corte:] do segundo;
        - mutação: com prob. 1%, inverte 1 bit aleatório da ação do filho;
        - o filho substitui um slot aleatório da ilha (pode ser um dos pais) e
          recebe budget = média dos budgets dos pais.
        """
        cfg = self.cfg
        if island is None:
            island = int(self.rng.integers(cfg.n_islands))
        p1, p2 = self._top_two(island)
        if self.rng.random() < 0.5:
            p1, p2 = p2, p1
        bits1 = half_to_bits(int(self.actions[island, p1]))
        bits2 = half_to_bits(int(self.actions[island, p2]))
        cut = int(self.rng.integers(1, N_BITS_HALF))               # 1, 2 ou 3
        child = bits1[:cut] + bits2[cut:]
        mutated = bool(self.rng.random() < cfg.mutation_rate)
        if mutated:
            pos = int(self.rng.integers(N_BITS_HALF))
            child[pos] ^= 1
        child_budget = float((self.budgets[island, p1] + self.budgets[island, p2]) / 2)
        target = int(self.rng.integers(cfg.actions_per_island))
        self.actions[island, target] = bits_to_half(child)
        self.budgets[island, target] = child_budget
        self.n_evolutions += 1
        return {"island": island, "parents": (p1, p2), "cut": cut, "mutated": mutated,
                "target": target, "child": bits_to_half(child), "child_budget": child_budget}
