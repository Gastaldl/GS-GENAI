"""Proteção de dados na pergunta e nos registros do assistente.

O CPF que aparecer na pergunta é mascarado antes da busca e do modelo (mask_cpf): ele nunca é
necessário para responder. Telefones e e-mails seguem para o modelo, porque a pergunta pode citar
um canal oficial, como o WhatsApp (11) 3651-7080 da Central 156.

Antes de gravar qualquer registro, CPF, telefone e e-mail que apareçam na pergunta são
mascarados. Cada registro guarda só o que a política de privacidade da base prevê
(base/documentos/produto-privacidade.md): a pergunta já mascarada (pergunta_mascarada), os
identificadores dos trechos recuperados para a resposta (trechos), a indicação de recusa
(recusa, com o detalhe em motivo_recusa: limiar, modelo, validacao ou dispositivo_sem_apoio) e o
tempo de resposta (latencia_s). Os registros ficam em um arquivo por dia, na pasta registros/ ou na indicada
pela variável de ambiente LUZ_LOG_DIR; a data está no nome do arquivo, e o registro não guarda
horário. Os arquivos com mais de 30 dias são apagados a cada gravação e quando a API é criada.
Não se grava IP, nome nem identificador de quem perguntou.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

from luz_segura.knowledge_base import ROOT

LOG_DIR = Path(os.environ.get("LUZ_LOG_DIR") or ROOT / "registros")
RETENTION_DAYS = 30
LOG_FIELDS = {"pergunta_mascarada", "trechos", "recusa", "motivo_recusa", "latencia_s"}

# CPF com ou sem separadores (espaço, ponto ou hífen): 123.456.789-09, 123 456 789 09, 12345678909
CPF = re.compile(r"\b\d{3}[\s.-]?\d{3}[\s.-]?\d{3}[\s.-]?\d{2}\b")
MASKS = [
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "[EMAIL]"),
    (CPF, "[CPF]"),
    # telefone com +55 e DDD opcionais, separado por espaço, hífen ou ponto: (11) 98888-7777, 3651.7080
    (re.compile(r"(\+?55\s?)?(\(?\d{2}\)?[\s.-]?)?9?\d{4}[\s.-]?\d{4}\b"), "[TELEFONE]"),
]


def mask_personal_data(text: str) -> str:
    for pattern, replacement in MASKS:
        text = pattern.sub(replacement, text)
    return text


def mask_cpf(text: str) -> str:
    """Mascara só o CPF: é assim que a pergunta chega à busca e ao modelo."""
    return CPF.sub("[CPF]", text)


def purge_old_logs(log_dir: Path | None = None, today: date | None = None) -> list[Path]:
    log_dir = log_dir or LOG_DIR
    today = today or date.today()
    removed = []
    for path in log_dir.glob("*.jsonl"):
        try:
            day = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if today - day > timedelta(days=RETENTION_DAYS):
            path.unlink()
            removed.append(path)
    return removed


def write_log(entry: dict, log_dir: Path | None = None) -> None:
    """entry deve trazer a pergunta já mascarada; só os campos previstos são gravados e a
    pergunta é mascarada de novo aqui, por garantia."""
    log_dir = log_dir or LOG_DIR
    record = {k: v for k, v in entry.items() if k in LOG_FIELDS}
    if "pergunta_mascarada" in record:
        record["pergunta_mascarada"] = mask_personal_data(record["pergunta_mascarada"])
    log_dir.mkdir(parents=True, exist_ok=True)
    purge_old_logs(log_dir)
    with open(log_dir / f"{date.today().isoformat()}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
