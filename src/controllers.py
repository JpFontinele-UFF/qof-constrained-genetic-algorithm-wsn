"""Controladores da PE executados no nó coordenador (CLAUDE.md, seções 4–5)."""
from __future__ import annotations

import numpy as np

from .classifier import ClassifierSystem, condition_from_delta
from .config import C1Config, GAConfig, SimConfig


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


class GAController(Controller):
    """Sistema classificador + AG original (seção 4), com critério de sucesso plugável.

    Ciclo a cada checkpoint t (chamada de update com Ef_t e QoF_t):
      1. avalia a regra aplicada em t−1 e, se bem-sucedida, ela recebe o banco;
      2. condição/ilha a partir de ΔEf = Ef_t − Ef_{t−1} (em t = 0, ΔEf = 0);
      3. sorteia uma regra da ilha (uniforme);
      4. cobra a regra, aplica a ação na PE (passando pelo hook `adjust_action`);
      5. a cada EVOL_PERIOD checkpoints (t % 2 == 1), executa uma evolução.
    """

    name = "GA"

    def __init__(self, ga_cfg: GAConfig | None = None):
        self.ga_cfg = ga_cfg or GAConfig()

    def reset(self, rng_ctrl, sim_cfg):
        super().reset(rng_ctrl, sim_cfg)
        self.cs = ClassifierSystem(rng_ctrl, self.ga_cfg)
        self.t = 0
        self.prev_ef: float | None = None
        self.prev_qof: float | None = None
        self.prev_rule: tuple[int, int] | None = None
        self.n_rewards = 0
        return self.pe

    # -- pontos de extensão
    def is_success(self, ef: float, qof: float) -> bool:
        raise NotImplementedError

    def adjust_action(self, dpe: float, qof: float) -> tuple[float, bool]:
        """Ação aplicada a partir da ação da regra. Original: sem alteração."""
        return dpe, False

    def remember(self, ef: float, qof: float) -> None:
        """Guarda as medidas de t para comparar em t+1."""
        self.prev_ef, self.prev_qof = ef, qof

    # -- ciclo
    def update(self, ef, qof):
        rewarded = False
        if self.prev_rule is not None:                              # 1. avaliar
            if self.is_success(ef, qof):
                self.cs.reward(*self.prev_rule)
                self.n_rewards += 1
                rewarded = True
        delta = 0.0 if self.prev_ef is None else ef - self.prev_ef  # 2. condição
        island = condition_from_delta(delta, self.ga_cfg.delta_ef_step)
        slot = self.cs.choose(island)                               # 3. escolha
        self.cs.charge(island, slot)                                # 4. cobrança + ação
        dpe_rule = self.cs.dpe(island, slot)
        dpe_applied, repaired, active = self._apply(dpe_rule, qof)
        self.pe = self.clip_pe(self.pe + dpe_applied)
        self.prev_rule = (island, slot)
        evolved = False
        if self.t % self.ga_cfg.evol_period == self.ga_cfg.evol_period - 1:  # 5. evolução
            self.cs.evolve()
            evolved = True
        self.remember(ef, qof)
        self.last_info = {"island": island, "slot": slot, "dpe_rule": dpe_rule,
                          "dpe_applied": dpe_applied, "repaired": repaired,
                          "repair_active": active, "rewarded": rewarded,
                          "evolved": evolved}
        self.t += 1
        return self.pe

    def _apply(self, dpe_rule: float, qof: float) -> tuple[float, bool, bool]:
        dpe, active = self.adjust_action(dpe_rule, qof)
        return dpe, (dpe != dpe_rule), active


class AbI(GAController):
    """Abordagem I (mono-objetivo): sucesso se Ef_t > Ef_{t−1}."""

    name = "AbI"

    def is_success(self, ef, qof):
        return ef > self.prev_ef


class AbII(GAController):
    """Abordagem II (multi-objetivo, nossa interpretação):
    se QoF_t >= QoF_MIN, sucesso se Ef_t > Ef_{t−1}; senão, sucesso se QoF_t > QoF_{t−1}.
    """

    name = "AbII"

    def is_success(self, ef, qof):
        if qof >= self.sim_cfg.qof_min:
            return ef > self.prev_ef
        return qof > self.prev_qof


class Proposed(AbII):
    """AbII + C1 (seção 5): penalidade e/ou reparo, com flags independentes.

    - use_penalty: sucesso se F_t > F_{t−1}, com
      F_t = Ef_t − LAMBDA · max(0, QoF_MIN − QoF_t) / QoF_MIN (substitui o critério da AbII);
    - use_repair: se QoF_t < QoF_MIN · (1 + MARGIN), a ação aplicada não reduz a PE;
      se QoF_t < QoF_MIN, aplica no mínimo ΔPE = +REPAIR_STEP. O cromossomo
      armazenado não é alterado.
    """

    def __init__(self, use_penalty: bool = True, use_repair: bool = True,
                 c1_cfg: C1Config | None = None, ga_cfg: GAConfig | None = None):
        super().__init__(ga_cfg)
        self.use_penalty = use_penalty
        self.use_repair = use_repair
        self.c1 = c1_cfg or C1Config()
        self.name = {(True, True): "AG-RQ", (True, False): "AbII+Pen",
                     (False, True): "AbII+Rep", (False, False): "AbII(Proposed)"}[
            (use_penalty, use_repair)]

    def reset(self, rng_ctrl, sim_cfg):
        pe = super().reset(rng_ctrl, sim_cfg)
        self.prev_f: float | None = None
        self.repairs = 0           # vezes em que o reparo mudou a ação aplicada
        self.repair_active = 0     # vezes em que a condição de reparo estava ativa
        return pe

    def fitness(self, ef: float, qof: float) -> float:
        qmin = self.sim_cfg.qof_min
        return ef - self.c1.lam * max(0.0, qmin - qof) / qmin

    def is_success(self, ef, qof):
        if self.use_penalty:
            return self.fitness(ef, qof) > self.prev_f
        return super().is_success(ef, qof)

    def remember(self, ef, qof):
        super().remember(ef, qof)
        self.prev_f = self.fitness(ef, qof)

    def adjust_action(self, dpe, qof):
        if not self.use_repair:
            return dpe, False
        qmin = self.sim_cfg.qof_min
        if qof < qmin:
            return max(dpe, self.c1.repair_step), True
        if qof < qmin * (1.0 + self.c1.margin):
            return max(dpe, 0.0), True
        return dpe, False

    def _apply(self, dpe_rule, qof):
        dpe, repaired, active = super()._apply(dpe_rule, qof)
        self.repairs += int(repaired)
        self.repair_active += int(active)
        return dpe, repaired, active
