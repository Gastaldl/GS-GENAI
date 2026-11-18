# Dados de treino do adaptador LoRA

O adaptador LoRA da configuração 3 (RAG + LoRA) aprende o comportamento do assistente diante dos trechos recuperados. Ele aprende a responder só com o que está nos trechos, citar os rótulos, responder a parte que consta, ignorar instruções embutidas na pergunta e recusar quando a informação não está nos trechos. **Todos os exemplos usam só os 35 trechos da partição de treino.** Trechos de dev e de teste nunca aparecem no treino.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `perguntas_treino.json` | material de partida: 175 perguntas com resposta (`qa`) e 80 perguntas sem resposta na base (`refusals`) |
| `build_training_data.py` | valida, descontamina e monta os exemplos (semente 2026) |
| `treino.jsonl` / `validacao.jsonl` | exemplos no formato prompt/completion do TRL, usados pelo notebook 03 |
| `composicao.json` | hashes, filtros, itens descartados e a composição de cada exemplo (tipo, pergunta, trechos e resposta esperada) |
| `convert_adapter.py` | converte o adaptador treinado (PEFT) para GGUF, o formato do llama.cpp, e copia as métricas do treino para `modelos/adaptador/metricas_treino.json` |
| `clean_colab_notebook.py` | tira do notebook 03 executado no Colab os metadados da conta Google, antes do commit |

Para gerar os exemplos de novo: `uv run python treino/build_training_data.py`.

## Material de partida

- **175 perguntas com resposta**, escritas a partir dos 35 trechos de treino em cinco estilos por trecho: direta, coloquial, procedimento, curta e com contexto pessoal. Cada resposta usa só o trecho de origem.
- **80 perguntas sem resposta na base**, cada uma com o motivo: 45 perto do tema (dados que a base não tem, outras cidades, protocolos de pessoas, previsões), 20 fora do tema e 15 pedidos impróprios ou tentativas de manipulação (revelar o prompt, mudar as regras, fingir ser atendente, expor dados de terceiros).

## Validação e descontaminação

1. **Respostas fiéis ao trecho.** O trecho de origem tem de ser de treino. Todo número citado na resposta precisa aparecer no trecho (título e seção incluídos) ou na própria pergunta. A resposta pode ter no máximo 4 frases e não pode citar rótulos de trecho no texto.
2. **Nenhuma pergunta parecida com as de avaliação.** Sai toda pergunta com cosseno ≥ 0,77 (bge-m3) ou Jaccard de trigramas de caracteres ≥ 0,45 com alguma pergunta de teste ou de dev. O limite de cosseno foi escolhido olhando a lista de pares ordenada: a partir de 0,77 apareciam perguntas com a mesma intenção de casos de avaliação, como o horário do WhatsApp do 156, a potência das lâmpadas LED ou a Copa de 2002.

Resultado: 158 das 175 perguntas com resposta e 57 das 80 sem resposta seguiram. As 40 descartadas, todas por semelhança com perguntas de avaliação, estão listadas em `composicao.json`. O notebook 01 também deixa essas 40 de fora ao escolher o limiar de similaridade.

A descontaminação compara perguntas, não fatos. Alguns fatos aparecem em mais de um documento da base (por exemplo, o telefone de emergência da Enel), e 12 casos de teste aceitam como evidência também um trecho de treino. O notebook 04 mostra o placar com e sem esses casos.

## Montagem dos exemplos

Cada exemplo segue o formato das mensagens da API, montado pela mesma função (`luz_segura.prompt.build_messages`). Tem o mesmo prompt de sistema (`SYSTEM_RAG`), a pergunta e três trechos rotulados `T1` a `T3`, como na busca densa usada no placar. No sistema final, a API às vezes acrescenta um 4º trecho (`T4`, o melhor do BM25), caso que o treino não cobre. A resposta esperada é o JSON `{"consta", "resposta", "fontes"}`. Antes da montagem, 10% das perguntas de cada estilo e de cada tipo de recusa foram separados para a validação. Assim, nenhuma pergunta aparece nos dois arquivos.

| tipo | o que ensina | treino | validação |
|---|---|---|---|
| `answer` | responder com o trecho certo entre 2 trechos parecidos (posição sorteada) e citar o rótulo | 142 | 16 |
| `two_chunks` | juntar duas respostas de documentos diferentes e citar os dois trechos | 36 | 4 |
| `partial` | responder a parte que consta e dizer que a outra parte não consta na base | 18 | 2 |
| `injection` | ignorar uma instrução embutida na pergunta (por exemplo, "esqueça o formato JSON") e responder normalmente | 27 | 3 |
| `removed` | recusar quando o trecho certo **não** está entre os recebidos; os 3 trechos vêm de outros documentos, sem os números da resposta e com texto pouco parecido com ela (cosseno < 0,60) | 54 | 6 |
| `refusal` | recusar perguntas sem resposta na base, com os 3 trechos de treino mais parecidos | 51 | 6 |
| **total** | | **328** (223 com resposta) | **37** (25 com resposta) |

Cerca de 12% das respostas esperadas no treino (38 de 328) começam com "Não". Dessas, 35 respondem "não" a uma pergunta de sim ou não ou negam o que a pergunta supõe. As outras 3 são orientações de segurança ("Não toque na pessoa...", "Não se aproxime e não toque no fio...").

Tamanho, em tokens do Qwen3: mediana de 1.037 e máximo de 1.545 no treino, abaixo do limite de 2.048 usado no treino.

## Treino

O treino roda no notebook `notebooks/03_treino_lora_colab.ipynb`, no Google Colab com GPU T4. A configuração é esta:

- QLoRA com o modelo base em 4 bits (nf4, quantização dupla) e cálculo em float16;
- pesos do LoRA em float32 durante o treino e salvos em float16 no zip. Com a base em 4 bits, o TRL 1.18 passa esses pesos para bfloat16 mesmo com `fp16=True` (problema em aberto no TRL, [huggingface/trl#7776](https://github.com/huggingface/trl/issues/7776)). A T4 não tem bfloat16, e o treino em float16 não aceita gradientes nesse tipo, então o notebook devolve os pesos para float32 logo depois de montar o `SFTTrainer`;
- LoRA de rank 16 e alpha 16, com dropout 0,05, nas 7 projeções (q, k, v, o, gate, up, down);
- 3 épocas;
- taxa de aprendizado 2e-4 com decaimento cosseno e 10 passos de aquecimento;
- lote efetivo 8 (1 × 8 acumulações);
- semente 42.

A perda conta só nos tokens da resposta, que é o padrão do TRL para dados prompt/completion. O notebook salva o adaptador de cada época, e a época usada é escolhida no dev (notebook 04).

## Resultado do treino

O treino rodou numa T4 do Colab em 46,6 minutos, com 123 passos e pico de 3,6 GiB de memória:

| época | perda na validação | placar líquido no dev (RAG + LoRA) |
|---|---|---|
| 1 | 0,237 | 0,900 |
| 2 | 0,244 | 0,925 |
| 3 | 0,254 | 0,925 |

A perda de treino caiu de 0,90 para 0,08, e a de validação subiu depois da 1ª época, sinal de que o adaptador passou a se ajustar demais aos exemplos. A escolha da época usa o placar no dev, e o empate entre as épocas 2 e 3 foi decidido pela menor perda de validação. Ficou a época 2.

O notebook 04 mostra um efeito colateral do treino. 72 das 223 respostas-alvo citam artigo de lei, e o adaptador passou a citar dispositivos com muito mais frequência. Uma vez, inventou um inciso (erro 5). Num próximo treino, convém incluir exemplos em que o trecho cita o artigo sem inciso e a resposta também não cita.

Depois do treino, todas as épocas do zip são convertidas para GGUF de uma vez. O script também copia do zip as métricas do treino e o exemplo formatado para `modelos/adaptador/metricas_treino.json` e `modelos/adaptador/exemplo_formatado.txt`. O zip tem cerca de 200 MB: deixe-o fora do repositório (por exemplo, na pasta Downloads) e passe o caminho dele:

```
uv run python treino/convert_adapter.py <caminho do adaptador_luz_segura.zip>
```

Para converter uma pasta PEFT avulsa: `uv run python treino/convert_adapter.py <pasta_da_época> modelos/epocas/epoca_<n>.gguf`.
