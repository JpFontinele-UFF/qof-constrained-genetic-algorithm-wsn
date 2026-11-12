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

## Sistema classificador e AG (Fase 2)

8. **Ordem dos bits.** Em cada metade do cromossomo (condição e ação), o bit 0 é o
   mais à esquerda e indica o sinal; os bits 1–3 são o nível em binário, com o
   bit 1 como mais significativo. Cada metade corresponde ao inteiro
   `sinal·8 + nível` (0..15), e o cromossomo de 8 bits é `condição·16 + ação`.
   A ilha de uma regra é o valor da sua condição (0..15).
9. **Faixas de |ΔEf|.** `nível = min(7, ⌊|ΔEf|/0,025⌋)`, com uma tolerância de
   1e-9 para que valores exatamente no limite da faixa (ex.: 0,075) não caiam na
   faixa inferior por erro de ponto flutuante. ΔEf = 0 tem sinal 0 ("caiu/igual").
10. **Ação com nível 0.** ΔPE = 0 qualquer que seja o bit de sinal.
11. **Cobrança e piso do budget.** A regra escolhida paga
    `min(100, budget − 1)` ao banco. Assim o piso de 1 é respeitado sem criar
    dinheiro, e a soma dos budgets com o banco é conservada exatamente entre
    evoluções. Uma regra no piso paga 0.
12. **Prêmio.** A regra bem-sucedida recebe todo o montante do banco, inclusive o
    que ela mesma pagou; o banco zera. Se a regra falha, o banco continua
    acumulando.
13. **Pais.** São as 2 regras de maior budget da ilha sorteada. Empates (por
    exemplo, no início, quando todas têm 1000) são desfeitos por sorteio. A ordem
    dos pais (quem fornece o início do filho) também é sorteada.
14. **Ponto de corte.** Uniforme em {1, 2, 3}, dentro dos 4 bits de ação: o filho
    herda os bits anteriores ao corte do primeiro pai e os demais do segundo. A
    condição não participa do cruzamento.
15. **Mutação.** Com probabilidade de 1% por filho, inverte 1 dos 4 bits de ação,
    escolhido ao acaso.
16. **Substituição.** O filho substitui um dos 4 slots da ilha, sorteado
    uniformemente (pode ser um dos pais), e recebe budget igual à média dos budgets
    dos pais. A população tem sempre 64 regras.
17. **Sorteios do controlador.** Ações iniciais, escolha da regra, desempates,
    ilha da evolução, corte, mutação e slot substituído usam o `rng_ctrl`.
