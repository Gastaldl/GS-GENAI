"""Cliente do llama-server (llama.cpp) que serve o modelo base e, opcionalmente, o adaptador LoRA.

O adaptador fica carregado no servidor e é ligado ou desligado a cada requisição pelo
campo "lora". O campo vai sempre explícito: sem ele, o servidor aplica a escala global
do adaptador (1,0), mesmo com --lora-init-without-apply. O cache de prompt fica
desligado para que a mesma pergunta gere sempre a mesma saída.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from luz_segura.knowledge_base import ROOT

# Variáveis de ambiente permitem apontar para outro servidor (por exemplo, no Docker)
# ou para o binário do llama.cpp em outro sistema operacional.
DEFAULT_URL = os.environ.get("LUZ_SERVER_URL", "http://127.0.0.1:8080")
BASE_MODEL_FILE = ROOT / "modelos" / "base" / "Qwen3-4B-Instruct-2507-Q8_0.gguf"
ADAPTER_FILE = ROOT / "modelos" / "adaptador" / "luz-segura-lora-f16.gguf"
SERVER_BINARY = Path(os.environ.get("LUZ_LLAMA_SERVER", ROOT / "ferramentas" / "llama-b12359-vulkan" / "llama-server.exe"))
SERVER_LOG = ROOT / ".cache" / "llama-server.log"
GENERATION = {"temperature": 0, "seed": 42, "max_tokens": 400}


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def server_command(model: Path = BASE_MODEL_FILE, adapter: Path | None = None, port: int = 8080,
                   binary: Path = SERVER_BINARY) -> list[str]:
    """Linha de comando do llama-server usada no projeto (um slot, contexto de 8192 tokens, tudo na GPU)."""
    command = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
               "-c", "8192", "-np", "1", "-ngl", "99"]
    if adapter is not None:
        command += ["--lora", str(adapter), "--lora-init-without-apply"]
    return command


def server_ready(base_url: str = DEFAULT_URL) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def start_server(adapter: Path | None = None, port: int = 8080, timeout: float = 300) -> subprocess.Popen:
    """Sobe o llama-server em segundo plano e espera o modelo carregar. O log fica em .cache/."""
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(SERVER_LOG, "w", encoding="utf-8")
    process = subprocess.Popen(server_command(adapter=adapter, port=port), stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"o llama-server terminou com código {process.returncode}; veja {SERVER_LOG}")
        if server_ready(url):
            return process
        time.sleep(1)
    process.terminate()
    raise TimeoutError(f"o llama-server não ficou pronto em {timeout:.0f} s; veja {SERVER_LOG}")


def file_id(path: Path) -> str:
    """Identificador rápido de um arquivo grande: tamanho + sha256 do primeiro e do último MiB."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(1 << 20))
        f.seek(max(0, size - (1 << 20)))
        digest.update(f.read(1 << 20))
    return digest.hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


class LlamaServer:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600,
                 transport: httpx.BaseTransport | None = None):
        """transport troca a rede por outro transporte do httpx (nos testes, um httpx.MockTransport).
        Com o servidor ainda carregando o modelo (503) ou uma resposta inesperada em /lora-adapters,
        levanta httpx.HTTPError, que a API devolve como 503, sem guardar um assistente quebrado."""
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout, transport=transport)
        try:
            self.adapters = self._lora_adapters()
        except Exception:
            self.http.close()
            raise

    def _lora_adapters(self) -> list[dict]:
        """Adaptadores LoRA carregados no servidor (lista vazia se ele subiu sem adaptador)."""
        response = self.http.get(f"{self.base_url}/lora-adapters")
        response.raise_for_status()
        try:
            adapters = response.json()
        except ValueError:  # corpo que não é JSON
            adapters = None
        if not isinstance(adapters, list):
            raise httpx.HTTPError("resposta inesperada de /lora-adapters")
        return adapters

    @property
    def has_adapter(self) -> bool:
        return bool(self.adapters)

    def properties(self) -> dict:
        response = self.http.get(f"{self.base_url}/props")
        response.raise_for_status()
        return response.json()

    def description(self) -> dict:
        """Modelo, quantização, versão do llama.cpp e adaptadores carregados."""
        props = self.properties()
        return {"model": Path(props.get("model_path", "")).name, "ftype": props.get("model_ftype"),
                "build": props.get("build_info"), "adapters": [Path(a["path"]).name for a in self.adapters]}

    def apply_template(self, messages: list[dict]) -> str:
        return self.http.post(f"{self.base_url}/apply-template", json={"messages": messages}).json()["prompt"]

    def chat(self, messages: list[dict], lora_scale: float = 0.0) -> Generation:
        body = {"messages": messages, "cache_prompt": False, **GENERATION}
        if self.adapters:
            body["lora"] = [{"id": a["id"], "scale": lora_scale} for a in self.adapters]
        elif lora_scale:
            raise RuntimeError("o servidor foi iniciado sem adaptador LoRA")
        start = time.perf_counter()
        response = self.http.post(f"{self.base_url}/v1/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return Generation(
            text=choice["message"]["content"] or "",
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            seconds=time.perf_counter() - start,
        )
