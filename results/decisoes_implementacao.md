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

## Ciclo do controlador, AbI e AbII (Fase 3)

18. **Primeiro checkpoint (t = 0).** Não há regra anterior para avaliar nem
    Ef_{−1}: usa-se ΔEf = 0, o que leva à ilha 0 ("caiu/igual", nível 0). A
    partir de t = 1, a regra aplicada em t−1 é avaliada normalmente.
19. **Ordem das operações no checkpoint t:** (1) avaliar e premiar a regra de
    t−1; (2) condição a partir de ΔEf; (3) sortear a regra da ilha; (4) cobrar a
    regra e aplicar a ação na PE (limitada a [PE_MIN, 1]); (5) evoluir, se for o
    caso. Como a evolução vem depois da escolha, a regra escolhida em t pode ser
    substituída pelo filho; nesse caso, o prêmio de t+1 vai para o slot (que já
    contém o filho).
20. **Momento da evolução.** Nos checkpoints t = 1, 3, 5, … (t % 2 == 1), ou
    seja, uma evolução a cada 2 checkpoints, 50 por execução.
21. **Critério da AbII com QoF_t = QoF_MIN.** Conta como "acima do limiar"
    (`QoF_t >= QoF_MIN`), e o sucesso é julgado pela Ef. Empate (Ef_t = Ef_{t−1}
    ou QoF_t = QoF_{t−1}) não é sucesso.
22. **Ação aplicada.** Na AbI e na AbII, a ação aplicada é a da regra; quando a
    PE é limitada em PE_MIN ou 1, a variação efetiva fica menor que a da regra.

### Comportamento observado: sementes presas com PE ≈ 1 (não corrigido)

Em algumas sementes (na AbI com N = 100: 20, 23 e 25, com PE média entre 0,77 e
0,99 nas sessões 10–99), a PE fica presa perto de 1. Isso decorre da própria
especificação original e **não foi corrigido**:

- com PE ≈ 1 a Ef quase não varia de uma sessão para outra (|ΔEf| < 0,025), de
  modo que só as duas ilhas de nível 0 (condições 0 e 8) são usadas;
- nessa região, uma regra que manda aumentar a PE não tem efeito (a PE já está no
  limite de 1), mas é premiada em cerca de metade das vezes, só pelo ruído da Ef;
- se a evolução eliminar as ações de redução dessas ilhas (na semente 23, a única
  ação de redução da ilha 0 foi substituída), a PE não sai mais de perto de 1;
- com escolha uniforme da regra e só cerca de 3 evoluções por ilha em 100 sessões,
  o sinal de aprendizado é fraco, e o comportamento se aproxima de um passeio
  aleatório da PE com aprendizado lento.

Consequência: a dispersão entre sementes é grande (PE média da AbI por semente
de 0,04 a 0,99), e a AbI e a AbII tomam decisões idênticas (mesma semente)
enquanto a QoF não cai abaixo de QoF_MIN.

## Contribuição C1 e oráculo (Fase 4)

23. **Penalidade no primeiro checkpoint.** Como na AbII, em t = 0 não há regra a
    avaliar; F_0 é guardado para a comparação em t = 1. Empate (F_t = F_{t−1})
    não é sucesso.
24. **Faixas do reparo** (Eq. 7 do artigo), avaliadas com a QoF da sessão que
    acabou de terminar: se QoF_t < QoF_MIN, ΔPE_ap = max(ΔPE, +0,06); se
    QoF_MIN ≤ QoF_t < 1,1·QoF_MIN, ΔPE_ap = max(ΔPE, 0); caso contrário,
    ΔPE_ap = ΔPE. QoF_t exatamente igual a QoF_MIN cai na faixa da margem. Depois
    do reparo, a PE ainda é limitada a [PE_MIN, 1].
25. **Contagem de reparos (`repairs`).** Conta as sessões em que o reparo mudou a
    ação aplicada (ΔPE_ap ≠ ΔPE da regra). À parte, registra-se `repair_active`:
    sessões em que a QoF estava numa das duas faixas do reparo, mesmo que a regra
    já mandasse aumentar a PE.
26. **Reparo e aprendizado.** O reparo altera só a ação aplicada. A regra
    escolhida paga ao banco normalmente e é avaliada na sessão seguinte pelo
    critério do método, mesmo que sua ação tenha sido trocada pelo reparo.
27. **Oráculo.** Usa o modelo sem interferência e o número esperado de
    transmissores k = N·PE (desigualdade de Jensen ignorada). Como o canal real
    tem 10% de sessões com interferência, a Ef* é um limite de referência, não
    exatamente alcançável. Com QoF_MIN fixo e problema viável, k* e Ef* não
    dependem de N (Ef* ≈ 0,484 para QoF_MIN = 12).
