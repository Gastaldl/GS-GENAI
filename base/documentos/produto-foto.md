---
id: produto-foto
title: Luz Segura, o que acontece com a foto e a localização enviadas
short_title: Luz Segura, tratamento da foto
publisher: Projeto Luz Segura (FIAP Global Solution 2026.2)
url: base/documentos/produto-foto.md
original_format: Markdown
kind: documento do produto
license: Texto do projeto
accessed: 2026-11-15
---

## Classificação automática da foto
Quando o relato chega, um modelo de visão computacional classifica a iluminação da foto em três níveis: escuro, parcialmente iluminado ou iluminado. Fotos que não mostram via pública, como fotos de documentos ou de ambientes internos, vão para revisão e não entram no cálculo de prioridade. Antes de guardar a imagem, o sistema apaga os metadados do arquivo, como modelo do celular e data exata, e desfoca rostos e placas de veículos detectados.

## Localização e tempo de guarda da foto
A localização enviada é arredondada para o trecho de rua, com precisão de cerca de 30 metros, antes de ser guardada. A foto fica guardada por até 90 dias ou até o ponto ser marcado como resolvido, o que acontecer primeiro; depois é apagada, e ficam só a classificação, o tipo de problema e a localização arredondada. As fotos não são publicadas: o mapa público mostra apenas pontos agregados.

## Verificação em campo com o robô
Quando um ponto acumula relatos, o robô de verificação, equipado com um sensor de luminosidade do tipo LDR, pode passar pelo local à noite e medir a luz. A medição confirma ou não a classificação das fotos e aumenta a confiança da prioridade. O sensor mede luminosidade relativa e não substitui a medição técnica de iluminância feita com luxímetro.
