"""Exemplos de treino do LoRA: formato, trechos só de treino e nenhuma pergunta de avaliação."""
import hashlib
import json
from pathlib import Path

from luz_segura.evaluation import load_cases, normalize_text
from luz_segura.prompt import SYSTEM_RAG
from luz_segura.split import load_split

TRAINING_DIR = Path(__file__).resolve().parents[1] / "treino"


def rows(name):
    with open(TRAINING_DIR / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_prompt_completion_format():
    for name in ("treino.jsonl", "validacao.jsonl"):
        for row in rows(name):
            assert set(row) == {"prompt", "completion"}
            system, user = row["prompt"]
            assert system == {"role": "system", "content": SYSTEM_RAG}
            assert user["role"] == "user" and user["content"].startswith("Trechos:\n\n[T1] ")
            target = json.loads(row["completion"][0]["content"])
            assert list(target) == ["consta", "resposta", "fontes"]
            assert target["consta"] == bool(target["resposta"]) == bool(target["fontes"])


def test_manifest_matches_files_and_uses_only_train_chunks():
    manifest = json.loads((TRAINING_DIR / "composicao.json").read_text(encoding="utf-8"))
    split = load_split()
    for name, info in manifest["files"].items():
        assert hashlib.sha256((TRAINING_DIR / name).read_bytes()).hexdigest() == info["sha256"]
        assert info["examples"] == len(rows(name))
    for example in manifest["examples"]:
        assert all(split[c] == "train" for c in example["context"]), example["question"]


def test_no_test_or_dev_question_in_training():
    manifest = json.loads((TRAINING_DIR / "composicao.json").read_text(encoding="utf-8"))
    training = {normalize_text(e["question"]) for e in manifest["examples"]}
    held_out = load_cases("casos_teste.jsonl") + load_cases("casos_dev.jsonl")
    assert not training & {normalize_text(c["question"]) for c in held_out}
