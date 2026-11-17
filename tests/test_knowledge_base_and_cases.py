"""Base de conhecimento, partição dos trechos e casos de avaliação."""
from collections import Counter

from luz_segura.evaluation import load_cases
from luz_segura.knowledge_base import MAX_WORDS, build_chunks, load_chunks, read_documents
from luz_segura.split import SEED, load_split, split_chunks


def test_saved_chunks_match_documents():
    assert build_chunks(read_documents()) == load_chunks()


def test_every_chunk_has_source_license_and_valid_size():
    for chunk in load_chunks():
        assert chunk.url and chunk.license and chunk.accessed, chunk.id
        assert 0 < chunk.word_count <= MAX_WORDS, chunk.id


def test_split_is_reproducible_and_covers_test():
    by_document = {}
    for chunk in load_chunks():
        by_document.setdefault(chunk.document, []).append(chunk.id)
    split = load_split()
    assert split_chunks(by_document, SEED) == split
    for document, ids in by_document.items():
        if len(ids) > 1:
            assert any(split[i] == "test" for i in ids), document


def test_cases_have_answer_key_literal_evidence_and_right_split():
    chunks = {c.id: c for c in load_chunks()}
    split = load_split()
    for name, partition in (("casos_teste.jsonl", "test"), ("casos_dev.jsonl", "dev")):
        for case in load_cases(name):
            assert case["split"] == partition
            for chunk_id, quote in case["evidence"]:
                assert quote in chunks[chunk_id].text, case["id"]
            if case["expected"] == "answer":
                assert case["key_points"] and case["gold_chunks"], case["id"]
                assert split[case["evidence"][0][0]] == partition, case["id"]
            else:
                assert not case["gold_chunks"], case["id"]


def test_evaluation_set_sizes():
    test = Counter(c["expected"] for c in load_cases("casos_teste.jsonl"))
    dev = Counter(c["expected"] for c in load_cases("casos_dev.jsonl"))
    assert test == {"answer": 39, "refuse": 21}
    assert dev == {"answer": 24, "refuse": 16}
