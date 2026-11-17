"""Mascaramento e retenção dos registros; aviso fixo de risco elétrico."""
import importlib
import json
from datetime import date, timedelta

import pytest

from luz_segura import privacy
from luz_segura.safety import indicates_electrical_danger


def test_masks_personal_data():
    text = "Sou o CPF 123.456.789-09, fone (11) 98888-7777, email maria.silva@exemplo.com.br"
    masked = privacy.mask_personal_data(text)
    assert "[CPF]" in masked and "[TELEFONE]" in masked and "[EMAIL]" in masked
    assert "123" not in masked and "98888" not in masked and "maria" not in masked


@pytest.mark.parametrize(("value", "mask"), [
    ("123.456.789-09", "[CPF]"),
    ("123.456.789.09", "[CPF]"),
    ("123 456 789 09", "[CPF]"),
    ("123 456 789-09", "[CPF]"),
    ("12345678909", "[CPF]"),
    ("(11) 98888-7777", "[TELEFONE]"),
    ("+55 11 98888 7777", "[TELEFONE]"),
    ("98888.7777", "[TELEFONE]"),
    ("3651.7080", "[TELEFONE]"),
    ("(11) 3651.7080", "[TELEFONE]"),
    ("maria.silva@exemplo.com.br", "[EMAIL]"),
])
def test_masks_each_format(value, mask):
    masked = privacy.mask_personal_data(f"Meu dado é {value}, obrigado.")
    assert masked == f"Meu dado é {mask}, obrigado."


def test_public_numbers_are_kept():
    text = "Liguei para a Enel no 0800 72 72 196 e para a Central 156; o CEP é 06010-000."
    assert privacy.mask_personal_data(text) == text


def test_log_keeps_only_allowed_fields_and_purges_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "LOG_DIR", tmp_path)
    old = tmp_path / f"{(date.today() - timedelta(days=31)).isoformat()}.jsonl"
    recent = tmp_path / f"{(date.today() - timedelta(days=29)).isoformat()}.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    recent.write_text("{}\n", encoding="utf-8")
    privacy.write_log({"pergunta_mascarada": "meu cpf 12345678909", "ip": "10.0.0.1", "configuracao": "rag",
                       "recusa": False})
    assert not old.exists() and recent.exists()
    saved = json.loads((tmp_path / f"{date.today().isoformat()}.jsonl").read_text(encoding="utf-8"))
    assert saved == {"pergunta_mascarada": "meu cpf [CPF]", "recusa": False}


def test_log_fields_follow_the_privacy_policy():
    """base/documentos/produto-privacidade.md: a pergunta já mascarada, os trechos, a indicação de
    recusa (com o motivo) e o tempo de resposta; nada de data e hora dentro do registro."""
    assert privacy.LOG_FIELDS == {"pergunta_mascarada", "trechos", "recusa", "motivo_recusa", "latencia_s"}


def test_log_dir_comes_from_luz_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LUZ_LOG_DIR", str(tmp_path))
    try:
        assert importlib.reload(privacy).LOG_DIR == tmp_path
    finally:
        monkeypatch.undo()
        importlib.reload(privacy)


@pytest.mark.parametrize("question", [
    "Tem um fio caído na calçada da minha rua",
    "o poste está inclinado e caindo",
    "meu filho levou um choque no portão",
    "saiu faísca do poste",
    "cabos soltos depois do vendaval",
    # sujeito antes do verbo
    "O fio caiu na frente de casa",
    "Os fios caíram depois da chuva",
    "Tem um cabo caindo do poste",
    "Um cabo caiu em cima do carro",
    "Tem um fio rompido na esquina",
    "O fio arrebentou com o vento",
    "Tem um cabo pendurado no poste",
    "Os fios estão pendurados na rua",
    "Arrebentou um fio aqui na rua",
    # postes no plural
    "Caíram dois postes na avenida",
    "Os postes caíram com o vendaval",
    # transformador
    "O transformador explodiu",
    "O transformador da esquina estourou",
    "O transformador está pegando fogo",
    "O transformador está soltando faísca",
    "Está saindo fumaça do transformador",
    "Ouvi uma explosão no transformador",
    "Tem fogo no poste",
    "Risco de eletrocução",
    # árvore ou galho em contato com a rede (ficha-risco-eletrico-02 e ficha-poda-02), nas duas ordens
    "Tem uma árvore caída em cima dos fios da rua",
    "Tem um galho encostado na fiação do poste",
    "Uma árvore está encostando nos fios da rede elétrica",
    "Os fios estão enroscados nos galhos da árvore",
    # fio encostado em alguma coisa
    "Tem um fio encostando no telhado da minha casa",
    # poste prestes a cair, balançando, rachado, tombado
    "O poste está prestes a cair",
    "Tem um poste balançando com o vento",
    "O poste da esquina está rachado",
    "O poste tombou na calçada",
    # perguntas de treino/perguntas_treino.json
    "Depois da chuva um fio caiu na calçada da minha rua e tem uma pessoa ferida perto. Quem eu chamo?",
    "Moro perto de uma praça e um fio caiu na calçada. Isso entra na fila de prioridade do Luz Segura?",
])
def test_detects_electrical_danger(question):
    assert indicates_electrical_danger(question)


@pytest.mark.parametrize("question", [
    "A lâmpada do poste da minha rua está apagada",
    "O poste da minha rua está com a luz piscando",
    "Quem conserta a lâmpada?",
    "Quanto custa a CIP?",
    "Como peço poda de árvore?",
    "A lâmpada do poste está quebrada",
    "A lâmpada do poste está rachada",
    "A lâmpada do poste estourou",
    "Liguei no 156 e a ligação caiu",
    # "caiu a energia", "caiu a chuva": falta de energia ou chuva, não algo que caiu
    "Quando caiu a energia o poste apagou",
    "O poste ficou sem luz quando caiu a chuva forte",
    "Caiu a força e o poste da rua apagou",
    # árvore perto do poste, sem fio na frase
    "Os galhos da árvore estão cobrindo a luz do poste",
])
def test_no_notice_without_danger(question):
    assert not indicates_electrical_danger(question)
