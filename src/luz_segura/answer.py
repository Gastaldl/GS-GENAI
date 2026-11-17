"""Leitura e validação da saída do modelo.

O modelo deve devolver só um objeto JSON com as chaves "consta", "resposta" e "fontes"
(a ordem das chaves não é conferida). A validação mede duas coisas:
- json_strict: o texto inteiro, sem nenhum ajuste, é um JSON nesse formato;
- usable: depois de remover espaços e uma eventual cerca de código (```json ... ```),
  o JSON está no formato e as fontes são rótulos válidos, sem repetição.
A API só aceita respostas "usable"; as demais viram a recusa padronizada.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

KEYS = ["consta", "resposta", "fontes"]
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


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
