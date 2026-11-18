"""Leitura e validação da saída do modelo.

O modelo deve devolver só um objeto JSON com as chaves "consta", "resposta" e "fontes"
(a ordem das chaves não é conferida). A validação mede duas coisas:
- json_strict: o texto inteiro, sem nenhum ajuste, é um JSON nesse formato;
- usable: depois de remover espaços e uma eventual cerca de código (```json ... ```),
  o JSON está no formato e as fontes são rótulos válidos, sem repetição.
A API só aceita respostas "usable"; as demais viram a recusa padronizada.

Além do formato, unsupported_provisions confere os incisos de lei citados na resposta ("art. 453,
inciso I") contra os trechos recebidos. A avaliação já confere os números da resposta contra o
material de apoio (evaluation.unsupported_data), mas um inciso em algarismo romano escapa dessa
conferência (notebook 04, erro 5).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

KEYS = ["consta", "resposta", "fontes"]
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

# Dispositivos citados na resposta: "art. 453, inciso I", "artigo 222, incisos I e III", "art. 150, I"
# (sem a palavra inciso, só com algarismo romano maiúsculo), "art. 7º, § 2º, IV" (inciso de um
# parágrafo) e "inciso V do art. 454".
ROMAN_LIST = r"[IVX]+(?:\s*(?:,|\be\b)\s*[IVX]+)*"
ARTICLE = r"(?i:\bart(?:igo)?\.?)\s*(\d+(?:-[A-Za-z])?)\s*[º°]?"
PARAGRAPH = r"(?:\s*,\s*§\s*\d+\s*[º°]?)?"
CITED_WITH_WORD = re.compile(ARTICLE + PARAGRAPH + r"\s*,\s*(?i:incisos?)\s+((?i:[ivx]+(?:\s*(?:,|\be\b)\s*[ivx]+)*))\b")
CITED_BARE = re.compile(ARTICLE + PARAGRAPH + r"\s*,\s*(" + ROMAN_LIST + r")\b")
CITED_REVERSED = re.compile(r"(?i:\bincisos?)\s+((?i:[ivx]+))\s+(?:(?i:do)\s+§\s*\d+\s*[º°]?\s+)?"
                            r"(?i:do\s+art(?:igo)?\.?)\s*(\d+(?:-[A-Za-z])?)")
ARTICLE_HEADING = re.compile(r"\bArt\.\s*(\d+(?:-[A-Z])?)\s*[.º°]")  # "Art. 453." e "Art. 1º" nos trechos de lei


def cited_incisos(answer: str) -> list[tuple[str, str]]:
    """Pares (artigo, inciso) citados na resposta, sem repetição: "art. 222, incisos I e III" dá
    [("222", "I"), ("222", "III")]."""
    found = []
    for pattern, article_group, items_group in ((CITED_WITH_WORD, 1, 2), (CITED_BARE, 1, 2), (CITED_REVERSED, 2, 1)):
        for match in pattern.finditer(answer):
            article = match.group(article_group).upper()
            for item in re.split(r"\s*(?:,|\be\b)\s*", match.group(items_group)):
                if item and (article, item.upper()) not in found:
                    found.append((article, item.upper()))
    return found


def _article_blocks(texts: list[str]) -> dict[str, list[str]]:
    """Texto de cada artigo nos trechos: do cabeçalho "Art. N." até o cabeçalho seguinte."""
    blocks: dict[str, list[str]] = {}
    for text in texts:
        heads = list(ARTICLE_HEADING.finditer(text))
        for i, head in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            blocks.setdefault(head.group(1).upper(), []).append(text[head.end():end])
    return blocks


def unsupported_provisions(answer: str, chunk_texts: list[str]) -> list[str]:
    """Incisos citados na resposta que os trechos não trazem. Um inciso tem apoio se o artigo aparece
    nos trechos com cabeçalho ("Art. 453.") e, no texto dele, o inciso como item ("I - "), ou se os
    trechos citam o mesmo artigo e inciso em prosa ("art. 150, I" ou "art. 7º, § 2º, IV")."""
    blocks = _article_blocks(chunk_texts)
    joined = "\n".join(chunk_texts)
    missing = []
    for article, item in cited_incisos(answer):
        as_item = re.compile(rf"(?<![A-Za-z]){item}\s*[-–—]\s")
        in_prose = re.compile(rf"(?i:\bart(?:igo)?\.?)\s*{re.escape(article)}\s*[º°]?\s*,\s*(?:§\s*\d+\s*[º°]?\s*,\s*)?"
                              rf"(?:(?i:incisos?)\s+)?(?:[IVX]+\s*(?:,|\be\b)\s*)*{item}\b", re.IGNORECASE)
        if not any(as_item.search(block) for block in blocks.get(article, [])) and not in_prose.search(joined):
            missing.append(f"art. {article}, inciso {item}")
    return missing


@dataclass(frozen=True)
class ParsedOutput:
    json_strict: bool
    usable: bool
    consta: bool | None
    resposta: str
    fontes: list
    error: str | None


def _schema_error(data) -> str | None:
    if not isinstance(data, dict):
        return "não é um objeto JSON"
    if sorted(data) != sorted(KEYS):
        return f"chaves diferentes de {KEYS}: {list(data)}"
    if not isinstance(data["consta"], bool):
        return "consta não é booleano"
    if not isinstance(data["resposta"], str):
        return "resposta não é texto"
    if not isinstance(data["fontes"], list) or not all(isinstance(f, str) for f in data["fontes"]):
        return "fontes não é lista de textos"
    return None


def parse_output(text: str, valid_labels: list[str], require_sources: bool) -> ParsedOutput:
    """valid_labels: rótulos dos trechos enviados (T1..Tk); require_sources: resposta com consta=true precisa citar."""
    strict = False
    try:
        strict = _schema_error(json.loads(text)) is None
    except (json.JSONDecodeError, TypeError):
        strict = False

    cleaned = text.strip()
    fence = FENCE.match(cleaned)
    if fence:
        cleaned = fence.group(1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return ParsedOutput(strict, False, None, "", [], f"JSON inválido: {exc.msg}")
    error = _schema_error(data)
    if error:
        return ParsedOutput(strict, False, None, "", [], error)

    consta, resposta, fontes = data["consta"], data["resposta"].strip(), data["fontes"]
    if not consta:
        return ParsedOutput(strict, True, False, "", [], None)
    if not resposta:
        return ParsedOutput(strict, False, True, "", fontes, "consta=true com resposta vazia")
    if len(set(fontes)) != len(fontes):
        return ParsedOutput(strict, False, True, resposta, fontes, "fontes repetidas")
    invalid = [f for f in fontes if f not in valid_labels]
    if invalid:
        return ParsedOutput(strict, False, True, resposta, fontes, f"rótulos inexistentes: {invalid}")
    if require_sources and not fontes:
        return ParsedOutput(strict, False, True, resposta, fontes, "resposta sem fonte")
    return ParsedOutput(strict, True, True, resposta, fontes, None)
