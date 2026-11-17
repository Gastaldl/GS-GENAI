"""Sorteio dos trechos entre treino, ajuste (dev) e teste.

As perguntas respondíveis herdam a partição do seu trecho principal (o primeiro trecho de
evidência do caso); as sem resposta, que não têm trecho de evidência, foram escritas direto
para dev ou teste. O adaptador LoRA só vê trechos de treino, o limiar e a escolha da época
usam o dev e o placar usa o teste.

Uso: python -m luz_segura.split   (gera avaliacao/particao_trechos.json)
"""
from __future__ import annotations

import hashlib
import json
import random

from luz_segura.knowledge_base import CHUNKS_FILE, ROOT, load_chunks

SPLIT_FILE = ROOT / "avaliacao" / "particao_trechos.json"
SEED = 2026
PROPORTIONS = {"train": 0.5, "dev": 0.2, "test": 0.3}


def _assign(ids: list[str], min_test: int) -> dict[str, str]:
    n_test = max(min_test, round(len(ids) * PROPORTIONS["test"]))
    n_dev = round(len(ids) * PROPORTIONS["dev"])
    return {
        chunk_id: "test" if position < n_test else "dev" if position < n_test + n_dev else "train"
        for position, chunk_id in enumerate(ids)
    }


def split_chunks(chunks_by_document: dict[str, list[str]], seed: int = SEED) -> dict[str, str]:
    """Sorteio estratificado por documento: todo documento com 2 ou mais trechos
    tem ao menos um trecho no teste; documentos de um trecho só são sorteados juntos."""
    rng = random.Random(seed)
    split, singles = {}, []
    for document in sorted(chunks_by_document):
        ids = sorted(chunks_by_document[document])
        if len(ids) == 1:
            singles.extend(ids)
            continue
        rng.shuffle(ids)
        split.update(_assign(ids, min_test=1))
    rng.shuffle(singles)
    split.update(_assign(singles, min_test=0))
    return dict(sorted(split.items()))


def load_split() -> dict[str, str]:
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["split"]


def main() -> None:
    by_document: dict[str, list[str]] = {}
    for chunk in load_chunks():
        by_document.setdefault(chunk.document, []).append(chunk.id)
    split = split_chunks(by_document)
    digest = hashlib.sha256(CHUNKS_FILE.read_bytes()).hexdigest()
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(
        {"seed": SEED, "proportions": PROPORTIONS, "chunks_sha256": digest, "split": split},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name in PROPORTIONS:
        ids = [i for i, s in split.items() if s == name]
        print(f"{name} ({len(ids)}): {', '.join(ids)}")


if __name__ == "__main__":
    main()
