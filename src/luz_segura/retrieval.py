"""Busca densa: embeddings multilíngues em CPU e similaridade de cosseno sobre os trechos.

O texto indexado de cada trecho inclui "documento > seção". Os vetores dos trechos ficam
em cache em .cache/, identificados pelo modelo (nome e commit fixado) e pelo conteúdo da base.
Cada modelo de embeddings é carregado num commit fixo do Hugging Face, sem a conversão
automática para safetensors, que baixaria uma segunda cópia de outro commit.

No fim do arquivo fica a busca corrigida (HybridRetriever), da análise de erros do
notebook 04: a busca densa com a segunda opinião de uma busca léxica (BM25).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from luz_segura.knowledge_base import ROOT, Chunk

CACHE_DIR = ROOT / ".cache" / "embeddings"


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    revision: str | None = None  # commit do modelo no Hugging Face (versão fixa)


EMBEDDING_MODELS = {
    spec.key: spec for spec in (
        EmbeddingSpec("e5-small", "intfloat/multilingual-e5-small", "query: ", "passage: ",
                      revision="614241f622f53c4eeff9890bdc4f31cfecc418b3"),
        EmbeddingSpec("granite-97m", "ibm-granite/granite-embedding-97m-multilingual-r2",
                      revision="835ad14087e140460703cf0fae09f97d469d65c2"),
        EmbeddingSpec("bge-m3", "BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181"),
    )
}


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk], spec: EmbeddingSpec):
        import torch
        from sentence_transformers import SentenceTransformer

        # sem isso, o transformers baixa em segundo plano uma cópia em safetensors de um PR de
        # conversão (outro commit), além do modelo no commit fixado
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        self.chunks = chunks
        self.spec = spec
        self.model = SentenceTransformer(
            spec.model_name, revision=spec.revision, device="cpu", model_kwargs={"torch_dtype": torch.float32}
        )
        self.matrix = self._chunk_vectors()

    @property
    def label(self) -> str:
        """Descrição da busca para o diagnóstico da API. A impressão digital das execuções com a
        busca densa usa evaluation.DENSE_RETRIEVAL, não este rótulo."""
        return self.spec.model_name

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype(np.float32)

    def _chunk_vectors(self) -> np.ndarray:
        texts = [self.spec.passage_prefix + c.indexed_text for c in self.chunks]
        model_id = f"{self.spec.model_name}@{self.spec.revision}"
        digest = hashlib.sha256((model_id + "\n" + "\n\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        cache_file = CACHE_DIR / f"{self.spec.key}-{digest}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        vectors = self._encode(texts)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, vectors)
        return vectors

    def query_vectors(self, questions: list[str]) -> np.ndarray:
        return self._encode([self.spec.query_prefix + q for q in questions])

    def scores(self, questions: list[str]) -> np.ndarray:
        """Matriz (perguntas x trechos) de similaridade de cosseno."""
        return self.query_vectors(questions) @ self.matrix.T

    def search(self, question: str, k: int = 3) -> list[Hit]:
        row = self.scores([question])[0]
        order = np.argsort(-row)[:k]
        return [Hit(self.chunks[i], float(row[i])) for i in order]


def load_retriever(chunks: list[Chunk], key: str) -> Retriever:
    return Retriever(chunks, EMBEDDING_MODELS[key])


# ---------------------------------------------------------------- busca léxica (correção do erro de busca)

STOPWORDS = set(
    "a o as os de da do das dos e em no na nos nas um uma uns umas que para por com sem se ao aos a as e eh ou mais "
    "como qual quais quem onde quando quanto quantos quantas meu minha minhas meus sua seu suas seus isso esse essa "
    "este esta tem ter ser foi sao pelo pela pelos pelas eu voce voces me te lhe ja nao sim tambem".split()
)
LEXICAL_PREFIX = 5  # radical simples: cada palavra fica com as 5 primeiras letras


def lexical_tokens(text: str, prefix: int | None = None) -> list[str]:
    """Palavras sem acento e sem palavras vazias; com prefix, cortadas nas primeiras letras."""
    from luz_segura.evaluation import normalize_text

    words = [w for w in re.findall(r"[a-z0-9]+", normalize_text(text)) if w not in STOPWORDS and len(w) > 1]
    return [w[:prefix] for w in words] if prefix else words


class LexicalIndex:
    """BM25 sobre o mesmo texto indexado da busca densa (documento > seção + texto)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75, prefix: int | None = None):
        self.prefix = prefix
        self.documents = [lexical_tokens(c.indexed_text, prefix) for c in chunks]
        self.frequencies = [Counter(d) for d in self.documents]
        self.average_length = sum(map(len, self.documents)) / len(self.documents)
        counts = Counter()
        for document in self.documents:
            counts.update(set(document))
        total = len(self.documents)
        self.idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in counts.items()}
        self.k1, self.b = k1, b

    def scores(self, question: str) -> np.ndarray:
        terms = lexical_tokens(question, self.prefix)
        out = np.zeros(len(self.documents))
        for i, (frequency, document) in enumerate(zip(self.frequencies, self.documents)):
            norm = self.k1 * (1 - self.b + self.b * len(document) / self.average_length)
            out[i] = sum(self.idf[t] * frequency[t] * (self.k1 + 1) / (frequency[t] + norm) for t in terms if t in frequency)
        return out


STRATEGIES = ("append", "replace")


class HybridRetriever:
    """Correção do erro de busca: a busca densa (bge-m3) continua escolhendo os trechos e a
    busca léxica (BM25) entra como segunda opinião, com uma de duas estratégias:

    - "append": os k trechos da busca densa e, se o melhor trecho do BM25 não estiver entre
      eles, ele entra também, como trecho k + 1 (nada do que a busca densa trazia sai);
    - "replace": os k - 1 primeiros da busca densa e, na última vaga, o melhor trecho do BM25
      que ainda não entrou (o k-ésimo da busca densa sai).

    Se o BM25 não acha nenhuma palavra da pergunta, fica só a busca densa. Nas duas estratégias
    o 1º trecho é o da busca densa, então a similaridade máxima e o limiar τ não mudam; a
    similaridade de cada Hit continua sendo o cosseno denso."""

    def __init__(self, dense: Retriever, prefix: int | None = LEXICAL_PREFIX, strategy: str = "append"):
        if strategy not in STRATEGIES:
            raise ValueError(f"estratégia desconhecida: {strategy}")
        self.dense = dense
        self.chunks = dense.chunks
        self.prefix = prefix
        self.strategy = strategy
        self.lexical = LexicalIndex(dense.chunks, prefix=prefix)

    @property
    def label(self) -> str:
        """Descrição da busca para o diagnóstico da API. Na impressão digital das execuções com
        esta busca, o notebook 04 usa este rótulo seguido do k (por exemplo, "... k=3")."""
        radical = f"radical de {self.prefix} letras" if self.prefix else "palavra inteira"
        return f"{self.dense.label} + BM25 ({radical}), estratégia {self.strategy}"

    def order(self, dense_row: np.ndarray, question: str, k: int = 3) -> list[int]:
        """Índices dos trechos escolhidos (k, ou k + 1 na estratégia "append"), na ordem do prompt."""
        dense_order = [int(i) for i in np.argsort(-dense_row)]
        lexical_row = self.lexical.scores(question)
        lexical_order = [int(i) for i in np.argsort(-lexical_row) if lexical_row[i] > 0]
        if self.strategy == "append":
            chosen = dense_order[:k]
            if lexical_order and lexical_order[0] not in chosen:
                chosen.append(lexical_order[0])
            return chosen
        chosen = dense_order[:k - 1]
        return chosen + [next((i for i in lexical_order if i not in chosen), dense_order[k - 1])]

    def search(self, question: str, k: int = 3) -> list[Hit]:
        dense_row = self.dense.scores([question])[0]
        return [Hit(self.chunks[i], float(dense_row[i])) for i in self.order(dense_row, question, k)]

    def scores(self, questions: list[str], k: int = 3) -> np.ndarray:
        """Pontuações com a mesma ordem de search: os escolhidos ficam na frente e o resto segue a
        ordem da busca densa. Na estratégia "append", o recall@k só vê os k primeiros; a cobertura
        de todos os trechos entregues sai de order()."""
        dense = self.dense.scores(questions)
        out = dense.astype(float) - 10.0  # cossenos ficam em [-1, 1]; os escolhidos recebem n, n-1, ..., 1
        for row, dense_row, question in zip(out, dense, questions):
            chosen = self.order(dense_row, question, k)
            for position, i in enumerate(chosen):
                row[i] = len(chosen) - position
        return out


def default_cache_dir() -> Path:
    return CACHE_DIR
