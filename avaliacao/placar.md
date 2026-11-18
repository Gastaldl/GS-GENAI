# Placar das três configurações (teste, 60 casos)

Gerado pelo notebook 04. Intervalos de Wilson (95%).

| configuração | placar líquido | acerto | corretas (respondíveis) | recusas corretas (sem resposta) | taxa de resposta | resposta sem apoio entre respondidas | JSON estrito | s por pergunta | cita trecho de evidência |
|---|---|---|---|---|---|---|---|---|---|
| sempre recusa | 0.35 | 21/60 = 35.0% (24.2 a 47.6) | 0/39 = 0.0% (0.0 a 9.0) | 21/21 = 100.0% (84.5 a 100.0) | 0/60 = 0.0% (0.0 a 6.0) | n/a |  |  |  |
| 1. sem contexto | 0.183 | 20/60 = 33.3% (22.7 a 45.9) | 1/39 = 2.6% (0.5 a 13.2) | 19/21 = 90.5% (71.1 a 97.3) | 12/60 = 20.0% (11.8 a 31.8) | 9/12 = 75.0% (46.8 a 91.1) | 60/60 = 100.0% (94.0 a 100.0) | 0.68 |  |
| 2. RAG simples | 0.783 | 49/60 = 81.7% (70.1 a 89.4) | 30/39 = 76.9% (61.7 a 87.4) | 19/21 = 90.5% (71.1 a 97.3) | 32/60 = 53.3% (40.9 a 65.4) | 2/32 = 6.2% (1.7 a 20.1) | 60/60 = 100.0% (94.0 a 100.0) | 2.16 | 30/30 = 100.0% (88.6 a 100.0) |
| 3. RAG + LoRA | 0.8 | 51/60 = 85.0% (73.9 a 91.9) | 31/39 = 79.5% (64.5 a 89.2) | 20/21 = 95.2% (77.3 a 99.2) | 35/60 = 58.3% (45.7 a 69.9) | 3/35 = 8.6% (3.0 a 22.4) | 60/60 = 100.0% (94.0 a 100.0) | 2.42 | 34/34 = 100.0% (89.8 a 100.0) |

## Desfechos

|  | 1. sem contexto | 2. RAG simples | 3. RAG + LoRA |
|---|---|---|---|
| CORRETA | 1 | 30 | 31 |
| INSUFICIENTE | 2 | 0 | 1 |
| ALUCINACAO | 7 | 0 | 2 |
| RECUSA_DISFARCADA | 0 | 0 | 0 |
| RECUSA_INDEVIDA | 29 | 9 | 5 |
| RECUSA_CORRETA | 19 | 19 | 20 |
| RESPOSTA_INDEVIDA | 2 | 2 | 1 |
