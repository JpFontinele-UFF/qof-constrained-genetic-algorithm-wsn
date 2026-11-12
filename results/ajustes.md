# Ajustes de parâmetros

Registro de qualquer alteração dos parâmetros da especificação (CLAUDE.md, versão 3).

## Parâmetros oficiais: nenhuma alteração

Todos os experimentos E1–E5 usam exatamente os valores da especificação:
`R_MAX = 23,52`, `K_HALF = 23,81`, `P_INTERF = 0,10`, `INTERF_DROP = 0,25`,
`LAMBDA = 2,0`, `MARGIN = 0,10`, `REPAIR_STEP = 0,06`, `QoF_MIN = 12` (padrão),
população 16 × 4, budget 1000, taxa 10%, evolução a cada 2 checkpoints, mutação 1%.

Nenhum parâmetro foi ajustado a partir dos resultados.

## E6 – Sensibilidade do reparo (análise definida a priori)

O E6 **não é uma alteração dos parâmetros oficiais**. É uma análise de
sensibilidade definida pelo autor em 12/11/2026, antes de qualquer número do E6
ser visto (depois da aprovação da Fase 4), para mostrar o compromisso entre
violação e eficiência do mecanismo de reparo.

- Método: AG-RQ, N = 100, QoF_MIN = 12, 30 sementes (0–29), LAMBDA = 2.
- Grade completa: MARGIN ∈ {0; 0,05; 0,10} × REPAIR_STEP ∈ {0,03; 0,06}
  (6 configurações), todas reportadas.
- Referências: PE100, AbII e AbII+Pen, nas mesmas sementes.
- Saídas: `results/e6_sensibilidade.{csv,tex}`, `results/e6_runs.csv`,
  `figuras/sensibilidade_tradeoff.png`.
- A configuração oficial (MARGIN = 0,10; REPAIR_STEP = 0,06) faz parte da grade e
  reproduz exatamente os números do AG-RQ no E2 (mesmas sementes).
- Os resultados do E2–E5 e as conclusões principais continuam baseados na
  configuração oficial.
