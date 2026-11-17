"""API do assistente Luz Segura.

Rotas:
- POST /perguntar  {"pergunta": "...", "configuracao": "rag_lora"}: resposta em JSON com fontes;
- GET  /saude: modelo carregado, limiar, configuração padrão e tamanho da base;
- GET  /: página de conversa, com o aviso de privacidade.
Se o llama-server estiver fora do ar ou falhar, /perguntar e /saude devolvem 503.

Variáveis de ambiente:
- LUZ_SERVER_URL: endereço do llama-server (padrão http://127.0.0.1:8080), lida em src/luz_segura/llm.py;
- LUZ_DEFAULT_MODE: configuração padrão, rag_lora, rag ou no_context. Sem ela, vale a escolhida no
  dev pelo notebook 04 (avaliacao/sistema_final.json) ou, na falta dela, rag_lora. Um valor fora
  dessa lista impede a criação do app. Sem adaptador no servidor, um modo com LoRA vira o RAG simples;
- LUZ_LOG_DIR: pasta dos registros diários (padrão registros/), lida em src/luz_segura/privacy.py.

Proteção de dados: a pergunta é processada só pela busca, dentro da API, e pelo llama-server de
LUZ_SERVER_URL (por padrão, na própria máquina). Ela só é guardada mascarada (CPF, telefone e
e-mail), num registro diário com os trechos recuperados, a recusa e o tempo de resposta, apagado
após 30 dias.

Uso (com o llama-server já rodando, veja o README):
    uv run uvicorn api.app:app --port 8000
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, get_args

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luz_segura.assistant import MODES, Assistant, load_final_system, load_search, load_threshold  # noqa: E402
from luz_segura.privacy import mask_personal_data, purge_old_logs, write_log  # noqa: E402

PAGE = Path(__file__).resolve().parent / "static" / "index.html"
ApiMode = Literal["rag_lora", "rag", "no_context"]
API_MODES: tuple[str, ...] = get_args(ApiMode)
FALLBACK_MODE = "rag_lora"
MODEL_UNAVAILABLE = "o modelo local não está disponível agora; confira se o llama-server está rodando"

logger = logging.getLogger(__name__)


class Question(BaseModel):
    pergunta: str = Field(min_length=3, max_length=500, description="Dúvida do morador, em português")
    configuracao: ApiMode | None = Field(
        default=None, description="Configuração do assistente; sem valor, usa a configuração padrão")


@contextmanager
def model_server_errors() -> Iterator[None]:
    """Falha de comunicação com o llama-server (fora do ar, conexão recusada, erro HTTP) vira 503,
    com a orientação de conferir o servidor; a causa fica no log da API."""
    try:
        yield
    except httpx.HTTPError as error:
        logger.warning("falha na comunicação com o llama-server: %s", error)
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE) from error


def build_assistant() -> Assistant:
    """Monta o assistente real. O llama-server vem primeiro: fora do ar, a falha aparece antes de
    carregar o modelo da busca."""
    from luz_segura.knowledge_base import load_chunks
    from luz_segura.llm import LlamaServer

    with model_server_errors():
        server = LlamaServer()
    chunks = load_chunks()
    return Assistant(chunks, load_search(chunks), server, threshold=load_threshold())


def mode_from_environment() -> str | None:
    """Configuração padrão pedida em LUZ_DEFAULT_MODE; None se a variável estiver vazia ou ausente."""
    value = os.environ.get("LUZ_DEFAULT_MODE", "").strip()
    if value and value not in API_MODES:
        accepted = ", ".join(API_MODES[:-1]) + " ou " + API_MODES[-1]
        raise ValueError(f"LUZ_DEFAULT_MODE={value!r} não é uma configuração da API; use {accepted}, "
                         "ou deixe a variável vazia")
    return value or None


def default_mode(assistant: Assistant, configured: str | None = None) -> str:
    """Configuração padrão: a de LUZ_DEFAULT_MODE (configured, já validada); senão, a escolhida no
    dev pelo notebook 04 (avaliacao/sistema_final.json), se for uma das aceitas pela API; senão,
    rag_lora. Sem adaptador no servidor, um modo com LoRA vira o RAG simples."""
    chosen = configured or load_final_system().get("mode")
    if chosen not in API_MODES:
        chosen = FALLBACK_MODE
    if MODES[chosen]["lora_scale"] and not assistant.server.has_adapter:
        return "rag"
    return chosen


def create_app(assistant: Assistant | None = None) -> FastAPI:
    """assistant=None monta o assistente real na primeira requisição (busca + llama-server).
    Ao criar o app, valida LUZ_DEFAULT_MODE (ValueError se inválida) e apaga os registros com
    mais de 30 dias."""
    configured = mode_from_environment()
    purge_old_logs()
    app = FastAPI(title="Luz Segura · assistente de iluminação pública", version="1.0.0")
    state: dict = {"assistant": assistant}

    def get_assistant() -> Assistant:
        if state["assistant"] is None:
            state["assistant"] = build_assistant()
        return state["assistant"]

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(PAGE)

    @app.get("/saude")
    def health() -> dict:
        current = get_assistant()
        with model_server_errors():
            model = current.server.description()
        return {"status": "ok", "modelo": model, "trechos": len(current.chunks),
                "busca": getattr(current.retriever, "label", None), "limiar": current.threshold,
                "configuracao_padrao": default_mode(current, configured)}

    @app.post("/perguntar")
    def ask(body: Question) -> dict:
        current = get_assistant()
        mode = body.configuracao or default_mode(current, configured)
        if MODES[mode]["lora_scale"] and not current.server.has_adapter:
            raise HTTPException(status_code=400, detail="o servidor do modelo foi iniciado sem o adaptador LoRA")
        with model_server_errors():
            result = current.run(body.pergunta, mode)
        write_log({
            "pergunta_mascarada": mask_personal_data(body.pergunta),
            "trechos": [h.chunk.id for h in result.hits],
            "recusa": result.refused,
            "motivo_recusa": result.refusal_reason,
            "latencia_s": round(result.seconds, 3),
        })
        return result.response

    return app


app = create_app()
