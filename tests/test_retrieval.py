"""Busca léxica (BM25) e as duas regras da busca corrigida: acrescentar o melhor trecho do BM25 como 4º
(a adotada no sistema final) ou trocar o 3º trecho da busca densa por ele."""
import numpy as np

from luz_segura.evaluation import markdown_table
from luz_segura.knowledge_base import load_chunks
from luz_segura.retrieval import HybridRetriever, Hit, LexicalIndex, lexical_tokens

CHUNKS = load_chunks()


def test_lexical_tokens_remove_accents_and_stopwords():
    assert lexical_tokens("Quantos caracteres tem o código do relato?") == ["caracteres", "codigo", "relato"]
    assert lexical_tokens("Quantos caracteres tem o código dos relatos?", prefix=5) == ["carac", "codig", "relat"]


def test_bm25_finds_the_chunk_with_the_rare_terms():
    index = LexicalIndex(CHUNKS)
    scores = index.scores("código de seis caracteres do relato")
    assert CHUNKS[int(np.argmax(scores))].id == "produto-relato-01"


class FakeDense:
    """Busca densa simulada: pontuação fixa para cada trecho, igual para qualquer pergunta."""

    def __init__(self, chunks, scores):
        self.chunks = chunks
        self.row = np.array(scores, dtype=np.float32)

    def scores(self, questions):
        return np.tile(self.row, (len(questions), 1))

    def search(self, question, k=3):
        return [Hit(self.chunks[i], float(self.row[i])) for i in np.argsort(-self.row)[:k]]


IDS = [c.id for c in CHUNKS]
QUESTION = "Quantos caracteres tem o código que o relato recebe?"


def dense_with_target_last():
    scores = np.linspace(0.9, 0.1, len(CHUNKS))
    scores[IDS.index("produto-relato-01")] = 0.05  # último na busca densa
    return scores


def test_hybrid_replace_keeps_the_dense_top_two_and_adds_the_best_lexical_match():
    dense_scores = dense_with_target_last()
    hybrid = HybridRetriever(FakeDense(CHUNKS, dense_scores), strategy="replace")
    hits = hybrid.search(QUESTION)
    assert [h.chunk.id for h in hits] == IDS[:2] + ["produto-relato-01"]
    assert max(h.score for h in hits) == np.float32(dense_scores[0])  # o limiar vê a mesma similaridade máxima
    order = np.argsort(-hybrid.scores([QUESTION])[0])[:3]
    assert [IDS[i] for i in order] == [h.chunk.id for h in hits]


def test_hybrid_append_keeps_the_dense_top_three_and_adds_the_best_lexical_match():
    dense_scores = dense_with_target_last()
    hybrid = HybridRetriever(FakeDense(CHUNKS, dense_scores), strategy="append")
    hits = hybrid.search(QUESTION)
    assert [h.chunk.id for h in hits] == IDS[:3] + ["produto-relato-01"]
    assert max(h.score for h in hits) == np.float32(dense_scores[0])


def test_hybrid_append_adds_nothing_when_the_lexical_best_is_already_there():
    dense_scores = np.linspace(0.1, 0.9, len(CHUNKS))
    dense_scores[IDS.index("produto-relato-01")] = 0.95  # 1º na busca densa
    hits = HybridRetriever(FakeDense(CHUNKS, dense_scores), strategy="append").search(QUESTION)
    assert len(hits) == 3 and hits[0].chunk.id == "produto-relato-01"


def test_hybrid_falls_back_to_dense_without_lexical_match():
    dense_scores = np.linspace(0.9, 0.1, len(CHUNKS))
    for strategy in ("append", "replace"):
        hits = HybridRetriever(FakeDense(CHUNKS, dense_scores), strategy=strategy).search("o que é isso?")
        assert [h.chunk.id for h in hits] == IDS[:3]


def test_markdown_table():
    import pandas as pd

    frame = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="linha"))
    assert markdown_table(frame) == "| linha | a |\n|---|---|\n| x | 1 |\n| y | 2 |"
