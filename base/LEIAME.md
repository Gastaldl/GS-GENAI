# Base de conhecimento

A base reúne o que o assistente pode usar para responder: atos oficiais, páginas oficiais e fichas sobre iluminação pública em Osasco, além dos documentos que descrevem o próprio Luz Segura. São 26 documentos e 76 trechos, com acesso a todas as fontes em 15/11/2026.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `documentos/*.md` | um arquivo por documento, com cabeçalho de metadados e uma seção `##` por trecho |
| `trechos.jsonl` | os 76 trechos gerados a partir dos documentos (gerado, não editar à mão) |
| `fontes.csv` e `fontes.md` | tabela de fontes: documento, fonte, órgão, endereço, formato original, tipo, licença, data de acesso, número de trechos, endereços próprios de seções e outras fontes consultadas (gerado) |
| `originais/` | cópias locais dos PDFs e HTMLs consultados, fora do repositório (`.gitignore`); os endereços estão em `fontes.csv` |

Para gerar `trechos.jsonl`, `fontes.csv` e `fontes.md` de novo a partir dos documentos:

```
uv run python -m luz_segura.knowledge_base
```

## Formato de um documento

```
---
id: lei13460
title: Lei nº 13.460, de 26 de junho de 2017 (...)
short_title: Lei 13.460/2017
publisher: Presidência da República (portal da legislação do Planalto)
url: https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2017/lei/l13460.htm
original_format: HTML (texto compilado)
kind: ato oficial, transcrição literal
license: Ato oficial sem proteção de direito autoral (Lei 9.610/1998, art. 8º, IV)
accessed: 2026-11-15
---

## Art. 7º: Carta de Serviços ao Usuário
<!-- url: endereço específico da seção, quando houver -->
Texto do trecho...
```

Cada seção `##` vira um trecho com id `<documento>-<número>` (por exemplo, `lei13460-02`). Um trecho tem no máximo 250 palavras. Na busca, o texto indexado é `título curto > seção`, seguido do texto. O endereço da seção, quando existe, fica no próprio trecho (campo `url` de `trechos.jsonl`), e a API o devolve junto com a resposta. A chave opcional `other_urls` do cabeçalho lista outras fontes consultadas para redigir uma ficha (por exemplo, o PDF da Lei municipal 3.995/2005 na ficha de poda); ela só entra na tabela de fontes.

## Tipos de documento e licenças

A forma como o texto entra na base depende do que a fonte permite.

| tipo | documentos | como o texto entra | base legal ou licença |
|---|---|---|---|
| ato oficial | Constituição (arts. 30 e 149-A), Lei 13.460/2017, REN ANEEL 1.000/2021, LC 404/2022, LC 448/2025, LC 464/2026 e Decreto 14.955/2025 de Osasco | transcrição literal | atos oficiais não têm proteção de direito autoral (Lei 9.610/1998, art. 8º, IV) |
| página gov.br | ANEEL (iluminação pública e canais das distribuidoras) e Ministério das Cidades (vendaval) | transcrição literal, sem alteração, com crédito | CC BY-ND 3.0 |
| publicação com permissão | Guia de PPP de Iluminação Pública do Banco Mundial (págs. 83 e 84) | trecho curto, literal, com a fonte | o guia permite reproduzir partes citando a fonte |
| ficha em redação própria | pedido na Central 156, prazos, ouvidoria, poda, risco elétrico, distribuidora, CIP, responsáveis e normas técnicas | fatos contados com palavras próprias, com endereço e data de acesso | as páginas de origem (Osasco Digital, Central 156, Enel, ABNT) não têm licença aberta ou proíbem cópia, por isso nada é copiado |
| documento do produto | o que é o Luz Segura, relato, foto, prioridade, privacidade e assistente | texto do projeto | texto do projeto |

A licença de cada documento está em `fontes.md`. Quando uma ficha resume um ato oficial (por exemplo, a ficha da CIP resume a LC 404/2022), a ficha está em linguagem simples, e o ato fica na base com o texto literal.

## Limites conhecidos

- A base cobre iluminação pública em Osasco e o funcionamento do Luz Segura. Ela não tem dados em tempo real (pedidos abertos, lâmpadas apagadas hoje), protocolos de pessoas, valores de tarifa de energia nem informações de outras cidades.
- Fatos de páginas que mudam com frequência (telefones, canais, situação da concessão da Enel) valem para a data de acesso, 15/11/2026.
- A edição de 2024 da ABNT NBR 5101 é paga e não foi consultada. A ficha de normas técnicas registra só o que o catálogo público informa.
- A REN ANEEL 1.000/2021 foi transcrita do texto compilado na plataforma de legislação indicada pela ANEEL (leis.org), com alterações até a REN 1.174/2026.
