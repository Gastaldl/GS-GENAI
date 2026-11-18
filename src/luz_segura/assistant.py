"""Fluxo do assistente: busca, limiar, geração, validação, fontes e aviso de segurança.

As três configurações comparadas usam o mesmo modelo base:
- no_context: só a pergunta, sem trechos e sem limiar;
- rag: os k trechos mais parecidos com a pergunta, com limiar de similaridade;
- rag_lora: igual ao rag, com o adaptador LoRA ligado.
O modo lora_no_context (adaptador ligado, sem trechos) é só um diagnóstico do notebook 04:
mostra o que o ajuste fino sabe sem a busca. Ele não faz parte do placar nem da API.

check_provisions=True recusa, nas configurações com trechos (rag e rag_lora), a resposta que cita
um inciso de lei ausente dos trechos (correção do erro 5 do notebook 04); a no_context, sem trechos,
não passa por essa conferência. As configurações do placar rodam sem ela; a API a liga se o
notebook 04 a adotou em avaliacao/sistema_final.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from luz_segura.answer import ParsedOutput, parse_output, unsupported_provisions
from luz_segura.knowledge_base import ROOT, Chunk
from luz_segura.llm import Generation, LlamaServer
from luz_segura.prompt import build_messages, chunk_label
from luz_segura.retrieval import Retriever
from luz_segura.safety import SAFETY_NOTICE, SAFETY_SOURCES, indicates_electrical_danger

MODES = {
    "no_context": {"context": False, "lora_scale": 0.0, "label": "sem contexto"},
    "rag": {"context": True, "lora_scale": 0.0, "label": "RAG simples"},
    "rag_lora": {"context": True, "lora_scale": 1.0, "label": "RAG + LoRA"},
    "lora_no_context": {"context": False, "lora_scale": 1.0, "label": "LoRA sem contexto"},
}
REFUSAL_MESSAGE = (
    "Não consta na base de conhecimento do Luz Segura. O assistente responde só a dúvidas sobre "
    "iluminação pública em Osasco, com base em documentos oficiais e nos documentos do projeto."
)
BELOW_THRESHOLD = "limiar"
MODEL_REFUSED = "modelo"
INVALID_OUTPUT = "validacao"
UNSUPPORTED_PROVISION = "dispositivo_sem_apoio"
THRESHOLD_FILE = ROOT / "avaliacao" / "limiar.json"
FINAL_SYSTEM_FILE = ROOT / "avaliacao" / "sistema_final.json"
MODEL_NAME = "Qwen3-4B-Instruct-2507"


def load_threshold() -> float:
    """Limiar de similaridade escolhido no notebook 01 (avaliacao/limiar.json)."""
    return float(json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))["threshold"])


def load_final_system() -> dict:
    """Configuração e busca escolhidas no dev pelo notebook 04 (avaliacao/sistema_final.json).
    Sem o arquivo, vale o sistema avaliado no placar: busca densa e escolha pela presença do adaptador."""
    if FINAL_SYSTEM_FILE.exists():
        return json.loads(FINAL_SYSTEM_FILE.read_text(encoding="utf-8"))
    return {"mode": None, "retrieval": {"adopted": False}}


def provision_check_adopted() -> bool:
    """True se o notebook 04 adotou a conferência dos incisos citados (avaliacao/sistema_final.json)."""
    return bool(load_final_system().get("provision_check", {}).get("adopted"))


def load_search(chunks: list[Chunk]):
    """Busca usada pela API: a densa (bge-m3) ou, se o notebook 04 adotou a correção da busca,
    a densa com a segunda opinião do BM25."""
    from luz_segura.retrieval import HybridRetriever, load_retriever

    dense = load_retriever(chunks, "bge-m3")
    settings = load_final_system()["retrieval"]
    if settings.get("adopted"):
        return HybridRetriever(dense, prefix=settings["prefix"], strategy=settings["strategy"])
    return dense


@dataclass
class AssistantResult:
    question: str
    mode: str
    hits: list = field(default_factory=list)
    max_score: float | None = None
    threshold: float | None = None
    generation: Generation | None = None
    parsed: ParsedOutput | None = None
    refusal_reason: str | None = None
    seconds: float = 0.0
    response: dict = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.refusal_reason is not None

    @property
    def cited_chunk_ids(self) -> list[str]:
        if self.refused or not self.parsed:
            return []
        by_label = {chunk_label(i): hit.chunk.id for i, hit in enumerate(self.hits)}
        return [by_label[label] for label in self.parsed.fontes if label in by_label]


class Assistant:
    def __init__(self, chunks: list[Chunk], retriever: Retriever | None, server: LlamaServer,
                 threshold: float, k: int = 3, model_name: str = MODEL_NAME, check_provisions: bool = False):
        self.chunks = {c.id: c for c in chunks}
        self.retriever = retriever
        self.server = server
        self.threshold = threshold
        self.k = k
        self.model_name = model_name
        self.check_provisions = check_provisions

    def run(self, question: str, mode: str, generate_below_threshold: bool = False) -> AssistantResult:
        """generate_below_threshold=True gera a resposta mesmo abaixo do limiar (usado na avaliação
        para medir o gerador isolado); a decisão final continua respeitando o limiar."""
        settings = MODES[mode]
        start = time.perf_counter()
        result = AssistantResult(question=question, mode=mode)
        chunks = None
        if settings["context"]:
            result.hits = self.retriever.search(question, self.k)
            result.max_score = max(hit.score for hit in result.hits)
            result.threshold = self.threshold
            chunks = [hit.chunk for hit in result.hits]
        below = settings["context"] and result.max_score < self.threshold
        if not below or generate_below_threshold:
            messages = build_messages(question, chunks)
            result.generation = self.server.chat(messages, lora_scale=settings["lora_scale"])
            labels = [chunk_label(i) for i in range(len(chunks or []))]
            result.parsed = parse_output(result.generation.text, labels, require_sources=settings["context"])
        if below:
            result.refusal_reason = BELOW_THRESHOLD
        elif not result.parsed.usable:
            result.refusal_reason = INVALID_OUTPUT
        elif not result.parsed.consta:
            result.refusal_reason = MODEL_REFUSED
        elif self.check_provisions and chunks and unsupported_provisions(result.parsed.resposta, [c.text for c in chunks]):
            result.refusal_reason = UNSUPPORTED_PROVISION
        result.seconds = time.perf_counter() - start
        result.response = self._response(result)
        return result

    def _response(self, result: AssistantResult) -> dict:
        diagnostics = {
            "modelo": f"{self.model_name} ({MODES[result.mode]['label']})",
            "similaridade_maxima": None if result.max_score is None else round(result.max_score, 4),
            "limiar": result.threshold,
            "motivo_recusa": result.refusal_reason,
        }
        if result.refused:
            body = {"consta": False, "resposta": REFUSAL_MESSAGE, "fontes": []}
        else:
            sources = []
            for chunk_id in result.cited_chunk_ids:
                chunk = self.chunks[chunk_id]
                sources.append({"id": chunk.id, "documento": chunk.document_title, "url": chunk.url, "trecho": chunk.text})
            body = {"consta": True, "resposta": result.parsed.resposta, "fontes": sources}
        notice = None
        if indicates_electrical_danger(result.question):
            notice = {"texto": SAFETY_NOTICE, "fontes": SAFETY_SOURCES}
        return {**body, "aviso_seguranca": notice, "diagnostico": diagnostics}


def result_record(result: AssistantResult) -> dict:
    """Registro completo de uma execução, para salvar as saídas brutas da avaliação."""
    return {
        "question": result.question, "mode": result.mode,
        "hits": [{"id": h.chunk.id, "score": round(h.score, 4)} for h in result.hits],
        "max_score": result.max_score, "threshold": result.threshold,
        "raw_output": result.generation.text if result.generation else None,
        "finish_reason": result.generation.finish_reason if result.generation else None,
        "prompt_tokens": result.generation.prompt_tokens if result.generation else None,
        "completion_tokens": result.generation.completion_tokens if result.generation else None,
        "generation_seconds": round(result.generation.seconds, 3) if result.generation else None,
        "json_strict": result.parsed.json_strict if result.parsed else None,
        "usable": result.parsed.usable if result.parsed else None,
        "model_consta": result.parsed.consta if result.parsed else None,
        "model_answer": result.parsed.resposta if result.parsed else None,
        "model_sources": result.parsed.fontes if result.parsed else None,
        "parse_error": result.parsed.error if result.parsed else None,
        "refusal_reason": result.refusal_reason,
        "cited_chunk_ids": result.cited_chunk_ids,
        "response": result.response,
        "seconds": round(result.seconds, 3),
    }
