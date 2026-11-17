"""Rede de segurança fixa para risco elétrico.

Se a pergunta indica perigo elétrico imediato (fio caído, choque, poste caindo, árvore encostada
nos fios, transformador explodindo...), a API sempre acrescenta o aviso abaixo, com fonte,
independentemente do que o modelo respondeu. É uma regra por palavras-chave, fora do placar das
três configurações. Ela roda sobre o texto em minúsculas e sem acentos (normalize_text) e aceita
o sujeito antes ou depois do verbo ("o fio caiu", "caiu um fio"), a até 30 caracteres um do outro,
na mesma frase. Perguntas comuns de iluminação ("a lâmpada do poste está apagada", "quando caiu
a energia o poste apagou") não levam o aviso.
"""
from __future__ import annotations

import re

from luz_segura.evaluation import normalize_text

NEAR = r"[^.?!]{0,30}"  # até 30 caracteres, sem passar para a frase seguinte
WIRE = r"\b(fios?|cabos?|fiacao|rede eletrica)\b"
EQUIPMENT = r"\b(fios?|cabos?|fiacao|postes?|transformador(es)?)\b"
TREE = r"\b(arvores?|galhos?)\b"
EXPLOSION = r"\b(explod|explosao|estour)"
# logo depois de "caiu": "caiu a energia", "caiu a luz", "caiu a força", "caiu a chuva" falam de
# falta de energia ou de chuva, não de algo que caiu ("quando caiu a energia o poste apagou" não leva o aviso)
NOT_OUTAGE = r"(?! a (energia|luz|forca|chuva))"

DANGER_PATTERNS = [
    # o fio caiu, os postes caíram, um cabo caindo (sujeito antes do verbo)
    EQUIPMENT + NEAR + r"\b(caiu|caindo|cairam)\b" + NOT_OUTAGE,
    # caiu um fio, caíram dois postes, arrebentou um cabo (verbo antes do sujeito)
    r"\b(caiu|caindo|cairam|arrebentou|arrebentaram|rompeu|romperam)\b" + NOT_OUTAGE + NEAR + EQUIPMENT,
    # fio caído, solto, rompido, arrebentado, pendurado, desencapado...
    WIRE + NEAR + (r"\b(caid[oa]s?|solt[oa]s?|partid[oa]s?|rompid[oa]s?|romp(eu|eram)|arrebent(ad[oa]s?|ou|aram)"
                   r"|pendurad[oa]s?|no chao|exposta?s?|desencapad[oa]s?|energizad[oa]s?)"),
    # fio encostando no telhado, cabo tocando no muro
    WIRE + NEAR + r"\b(encost|tocando|tocou)",
    # árvore ou galho em contato com a rede, nas duas ordens (base: ficha-risco-eletrico-02 e
    # ficha-poda-02); sem fio na frase, como em "como peço poda de árvore?", não leva o aviso
    TREE + NEAR + WIRE,
    WIRE + NEAR + TREE,
    # poste caído, tombando, torto, rachado, balançando, prestes a cair... os adjetivos só no
    # masculino, para não pegar "a lâmpada do poste está quebrada"
    r"\bpostes?\b" + NEAR + (r"\b(caido|tomb(ando|ou|aram)|torto|inclinado|abalroado|quebrado|rachado"
                             r"|balancando|prestes a cair)"),
    # transformador que explodiu ou estourou, nas duas ordens ("a lâmpada estourou" não leva o aviso)
    r"\btransformador(es)?\b" + NEAR + EXPLOSION,
    EXPLOSION + NEAR + r"\btransformador",
    # fogo ou fumaça em fio, poste ou transformador, nas duas ordens
    r"\b(fios?|cabos?|fiacao|rede eletrica|postes?|transformador(es)?)\b" + NEAR
    + r"\b(pegando fogo|pegou fogo|em chamas|fumaca)",
    r"\b(fogo|fumaca)\b" + NEAR + EQUIPMENT,
    r"\bchoque( eletrico)?\b",
    r"\beletrocu",  # eletrocutado, eletrocução
    r"\bfaisca",
    r"\bcurto[- ]circuito\b",
]

SAFETY_NOTICE = (
    "Atenção: se houver fio caído, poste danificado ou risco de choque, não se aproxime e não toque em fios "
    "nem em objetos encostados neles. Ligue para a Emergência da Enel, 0800 72 72 196 (24 horas). Se houver "
    "vítima ou incêndio, ligue também para o SAMU (192) ou para o Corpo de Bombeiros (193)."
)
SAFETY_SOURCES = ["aneel-distribuidoras-01", "ficha-risco-eletrico-01", "ficha-risco-eletrico-04"]


def indicates_electrical_danger(question: str) -> bool:
    text = normalize_text(question)
    return any(re.search(pattern, text) for pattern in DANGER_PATTERNS)
