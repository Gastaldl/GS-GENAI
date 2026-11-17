"""API com busca e servidor do modelo simulados (não precisa de GPU nem de rede)."""
import json
from datetime import date, timedelta
from functools import partial

import httpx
import pytest
from fastapi.testclient import TestClient

import api.app
from api.app import create_app
from luz_segura import privacy
from luz_segura.assistant import REFUSAL_MESSAGE, Assistant
from luz_segura.knowledge_base import load_chunks
from luz_segura.llm import Generation, LlamaServer
from luz_segura.retrieval import Hit

CHUNKS = load_chunks()
MODEL_UNAVAILABLE = "o modelo local não está disponível agora; confira se o llama-server está rodando"


class FakeRetriever:
    def __init__(self, ids, score):
        self.hits = [next(c for c in CHUNKS if c.id == i) for i in ids]
        self.score = score

    def search(self, question, k=3):
        return [Hit(chunk, self.score - 0.01 * n) for n, chunk in enumerate(self.hits[:k])]


class FakeServer:
    def __init__(self, output, has_adapter=False, error=None):
        self.output = output
        self.has_adapter = has_adapter
        self.adapters = [{"id": 0}] if has_adapter else []
        self.error = error
        self.calls = []

    def chat(self, messages, lora_scale=0.0):
        self.calls.append(lora_scale)
        if self.error is not None:
            raise self.error
        return Generation(self.output, "stop", 900, 30, 0.5)

    def description(self):
        return {"model": "simulado.gguf", "ftype": "Q8_0", "build": "teste", "adapters": []}


def client_for(output, score=0.7, has_adapter=False, error=None):
    retriever = FakeRetriever(["ficha-pedido-156-02", "ren1000-02", "ficha-risco-eletrico-01"], score)
    assistant = Assistant(CHUNKS, retriever, FakeServer(output, has_adapter, error), threshold=0.42)
    return TestClient(create_app(assistant)), assistant


@pytest.fixture(autouse=True)
def temporary_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "LOG_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def final_system(monkeypatch):
    """Os testes não dependem de avaliacao/sistema_final.json: cada um escolhe o sistema final."""
    chosen = {"mode": None, "retrieval": {"adopted": False}}
    monkeypatch.setattr(api.app, "load_final_system", lambda: chosen)
    monkeypatch.delenv("LUZ_DEFAULT_MODE", raising=False)
    return chosen


def test_answer_with_sources_and_diagnostics():
    client, _ = client_for('{"consta": true, "resposta": "Ligue para o 156.", "fontes": ["T1"]}')
    data = client.post("/perguntar", json={"pergunta": "Como peço conserto da lâmpada?"}).json()
    assert data["consta"] is True and data["resposta"] == "Ligue para o 156."
    assert [f["id"] for f in data["fontes"]] == ["ficha-pedido-156-02"]
    assert data["fontes"][0]["url"].startswith("https://")
    assert data["aviso_seguranca"] is None
    assert data["diagnostico"]["limiar"] == 0.42 and data["diagnostico"]["motivo_recusa"] is None


def test_refuses_below_threshold_without_calling_model():
    client, assistant = client_for('{"consta": true, "resposta": "x", "fontes": ["T1"]}', score=0.30)
    data = client.post("/perguntar", json={"pergunta": "Qual é a capital da Austrália?"}).json()
    assert data == {**data, "consta": False, "resposta": REFUSAL_MESSAGE, "fontes": []}
    assert data["diagnostico"]["motivo_recusa"] == "limiar"
    assert assistant.server.calls == []


def test_invalid_output_becomes_refusal():
    client, _ = client_for("Ligue para o 156, é rapidinho!")
    data = client.post("/perguntar", json={"pergunta": "Como peço conserto da lâmpada?"}).json()
    assert data["consta"] is False and data["diagnostico"]["motivo_recusa"] == "validacao"


def test_safety_notice_and_masked_log(temporary_logs):
    client, _ = client_for('{"consta": false, "resposta": "", "fontes": []}')
    data = client.post("/perguntar", json={"pergunta": "Fio caído aqui! Meu telefone é (11) 98888-7777"}).json()
    assert data["consta"] is False
    assert "0800 72 72 196" in data["aviso_seguranca"]["texto"]
    saved = json.loads((temporary_logs / f"{date.today().isoformat()}.jsonl").read_text(encoding="utf-8"))
    assert "98888" not in saved["pergunta_mascarada"] and "[TELEFONE]" in saved["pergunta_mascarada"]
    assert set(saved) == {"pergunta_mascarada", "trechos", "recusa", "motivo_recusa", "latencia_s"}
    assert saved["recusa"] is True and saved["motivo_recusa"] == "modelo"


def test_configurations_and_input_validation():
    client, assistant = client_for('{"consta": true, "resposta": "Ligue para o 156.", "fontes": []}')
    assert client.post("/perguntar", json={"pergunta": "Oi?", "configuracao": "no_context"}).json()["consta"] is True
    assert client.post("/perguntar", json={"pergunta": "Quem conserta?", "configuracao": "rag_lora"}).status_code == 400
    assert client.post("/perguntar", json={"pergunta": "a"}).status_code == 422
    assert client.get("/").status_code == 200
    assert client.get("/saude").json()["configuracao_padrao"] == "rag"
    lora_client, lora_assistant = client_for('{"consta": false, "resposta": "", "fontes": []}', has_adapter=True)
    lora_client.post("/perguntar", json={"pergunta": "Quem conserta a lâmpada?"})
    assert lora_assistant.server.calls == [1.0]


def test_default_configuration_follows_the_final_system(final_system):
    final_system["mode"] = "rag"
    client, assistant = client_for('{"consta": false, "resposta": "", "fontes": []}', has_adapter=True)
    client.post("/perguntar", json={"pergunta": "Quem conserta a lâmpada?"})
    assert assistant.server.calls == [0.0]
    final_system["mode"] = "rag_lora"
    client, assistant = client_for('{"consta": false, "resposta": "", "fontes": []}', has_adapter=False)
    assert client.get("/saude").json()["configuracao_padrao"] == "rag"  # sem adaptador no servidor


@pytest.mark.parametrize("chosen", ["lora_no_context", "turbo"])
def test_invalid_final_system_mode_is_ignored(final_system, chosen):
    final_system["mode"] = chosen
    client, _ = client_for('{"consta": false, "resposta": "", "fontes": []}', has_adapter=True)
    assert client.get("/saude").json()["configuracao_padrao"] == "rag_lora"


def test_default_mode_variable_overrides_the_final_system(final_system, monkeypatch):
    final_system["mode"] = "rag"
    monkeypatch.setenv("LUZ_DEFAULT_MODE", "no_context")
    client, _ = client_for('{"consta": false, "resposta": "", "fontes": []}', has_adapter=True)
    assert client.get("/saude").json()["configuracao_padrao"] == "no_context"
    monkeypatch.setenv("LUZ_DEFAULT_MODE", "rag_lora")
    client, _ = client_for('{"consta": false, "resposta": "", "fontes": []}', has_adapter=False)
    assert client.get("/saude").json()["configuracao_padrao"] == "rag"  # sem adaptador no servidor


@pytest.mark.parametrize("value", ["turbo", "lora_no_context"])
def test_invalid_default_mode_variable_fails_when_creating_the_app(monkeypatch, value):
    monkeypatch.setenv("LUZ_DEFAULT_MODE", value)
    with pytest.raises(ValueError, match="LUZ_DEFAULT_MODE"):
        client_for('{"consta": false, "resposta": "", "fontes": []}')


def test_unreachable_model_server_returns_503():
    client, _ = client_for("", error=httpx.ConnectError("conexão recusada"))
    response = client.post("/perguntar", json={"pergunta": "Como peço conserto da lâmpada?"})
    assert response.status_code == 503
    assert response.json() == {"detail": MODEL_UNAVAILABLE}


def test_model_server_down_while_building_the_assistant_returns_503(monkeypatch):
    class OfflineServer:
        def __init__(self, *args, **kwargs):
            raise httpx.ConnectError("conexão recusada")

    monkeypatch.setattr("luz_segura.llm.LlamaServer", OfflineServer)
    client = TestClient(create_app())
    for response in (client.post("/perguntar", json={"pergunta": "Quem conserta a lâmpada?"}), client.get("/saude")):
        assert response.status_code == 503
        assert response.json() == {"detail": MODEL_UNAVAILABLE}


def loading_model(request: httpx.Request) -> httpx.Response:
    """Resposta do llama-server enquanto o modelo ainda está carregando."""
    return httpx.Response(503, json={"error": {"code": 503, "message": "Loading model"}})


def test_llama_server_raises_while_the_model_is_loading():
    with pytest.raises(httpx.HTTPStatusError) as error:
        LlamaServer(transport=httpx.MockTransport(loading_model))
    assert error.value.response.status_code == 503


def test_llama_server_rejects_an_unexpected_lora_adapters_response():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"adapters": []}))
    with pytest.raises(httpx.HTTPError, match="/lora-adapters"):
        LlamaServer(transport=transport)


def test_model_still_loading_returns_503_until_the_server_is_ready(monkeypatch):
    """Durante a carga, cada requisição tenta montar o assistente de novo e recebe 503, sem guardar
    um assistente quebrado; quando o servidor fica pronto, a API passa a responder."""
    server = {"loading": True}
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if server["loading"]:
            return loading_model(request)
        if request.url.path == "/lora-adapters":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"model_path": "modelos/base/simulado.gguf", "model_ftype": "Q8_0",
                                         "build_info": "teste"})

    monkeypatch.setattr("luz_segura.llm.LlamaServer", partial(LlamaServer, transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(api.app, "load_search", lambda chunks: None)  # sem carregar a busca real (bge-m3)
    client = TestClient(create_app())
    for response in (client.post("/perguntar", json={"pergunta": "Quem conserta a lâmpada?"}), client.get("/saude")):
        assert response.status_code == 503
        assert response.json() == {"detail": MODEL_UNAVAILABLE}
    server["loading"] = False
    response = client.get("/saude")
    assert response.status_code == 200 and response.json()["modelo"]["model"] == "simulado.gguf"
    assert paths == ["/lora-adapters", "/lora-adapters", "/lora-adapters", "/props"]


def test_old_logs_are_purged_when_the_app_is_created(temporary_logs):
    old = temporary_logs / f"{(date.today() - timedelta(days=31)).isoformat()}.jsonl"
    recent = temporary_logs / f"{(date.today() - timedelta(days=29)).isoformat()}.jsonl"
    for path in (old, recent):
        path.write_text("{}\n", encoding="utf-8")
    client_for('{"consta": false, "resposta": "", "fontes": []}')
    assert not old.exists() and recent.exists()
