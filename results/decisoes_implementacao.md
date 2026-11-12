# Decisões de implementação

Decisões tomadas na implementação do simulador que **não estão especificadas** no
CLAUDE.md (versão 3). Todas foram aprovadas antes da implementação e valem
igualmente para todos os métodos comparados. Mudanças de parâmetros da
especificação (se houver) ficam em `results/ajustes.md`, não aqui.

## Simulador e canal (Fase 1)

1. **Rodada de overhead.** Das 20 rodadas de cada sessão, as 19 primeiras são
   monitoradas e a 20ª é o checkpoint: nela não há transmissões, e ela não entra
   nas mensagens enviadas nem nas recebidas. Logo, `QoF_t = recebidas_t / 19`.
2. **Números aleatórios.** Para cada semente `s`: `rng_channel = default_rng(s)` e
   `rng_ctrl = default_rng(s + 10000)`. As 100 sessões com/sem interferência são
   sorteadas primeiro, com `rng_channel`; em seguida, o mesmo `rng_channel` sorteia
   `k` e `r` de cada rodada. Como `k` depende da PE escolhida, apenas a sequência de
   interferências é idêntica entre os métodos (é o que a especificação exige); os
   sorteios de transmissão divergem assim que as PEs divergem.
3. **Sorteio por sessão.** Os `k` das 19 rodadas são sorteados de uma vez
   (`Binomial(N, PE)` com 19 valores) e, em seguida, `r ~ Binomial(k, s(k))` para
   cada rodada. Isso é equivalente a sortear rodada a rodada.
4. **Proteção em s(k).** Usa-se `s(k) = min(1, R_eff/(K_HALF + k))`. Com os
   parâmetros calibrados (`R_MAX < K_HALF`) o `min` nunca atua; é só uma proteção.
5. **Eficiência de várias sessões ou sementes.** É calculada como razão de totais
   (recebidas totais / enviadas totais), e não como média das razões.
6. **Limites da PE.** A PE devolvida pelo controlador é sempre limitada a
   `[PE_MIN, 1]` pelo simulador (além da limitação feita no próprio controlador).
7. **Cenário dinâmico (E5).** O N vale 100 nas sessões 0–49 e 200 nas sessões 50–99
   (sessões numeradas a partir de 0).
