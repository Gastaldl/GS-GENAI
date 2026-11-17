"""Contrato de saída do modelo, classificação das respostas, métricas e impressão digital das execuções."""
import json

import pytest

from luz_segura import evaluation
from luz_segura.answer import parse_output
from luz_segura.evaluation import (CORRECT, CORRECT_REFUSAL, DISGUISED_REFUSAL, HALLUCINATION, INSUFFICIENT,
                                   WRONG_ANSWER, WRONG_REFUSAL, configuration_fingerprint, judge, load_cases, mcnemar,
                                   summarize, unsupported_data, wilson)
from luz_segura.knowledge_base import load_chunks
from luz_segura.prompt import SYSTEM_NO_CONTEXT, SYSTEM_RAG, build_messages, chunk_label

LABELS = ["T1", "T2", "T3"]
CHUNKS = {c.id: c for c in load_chunks()}
CASES = {c["id"]: c for c in load_cases("casos_teste.jsonl")}


@pytest.mark.parametrize("text, strict, usable, error_part", [
    ('{"consta": true, "resposta": "Ligue 156.", "fontes": ["T1"]}', True, True, None),
    ('```json\n{"consta": true, "resposta": "Ligue 156.", "fontes": ["T1"]}\n```', False, True, None),
    ('{"consta": false, "resposta": "", "fontes": []}', True, True, None),
    ('{"consta": true, "resposta": "Ligue 156.", "fontes": ["T9"]}', True, False, "inexistentes"),
    ('{"consta": true, "resposta": "Ligue 156.", "fontes": []}', True, False, "sem fonte"),
    ('{"consta": true, "resposta": "Ligue 156.", "fontes": ["T1", "T1"]}', True, False, "repetidas"),
    ('{"consta": true, "resposta": "", "fontes": ["T1"]}', True, False, "vazia"),
    ('{"consta": true, "texto": "Ligue 156."}', False, False, "chaves"),
    ("Ligue para o 156.", False, False, "JSON inválido"),
])
def test_output_validation(text, strict, usable, error_part):
    parsed = parse_output(text, LABELS, require_sources=True)
    assert parsed.json_strict is strict
    assert parsed.usable is usable
    if error_part:
        assert error_part in parsed.error


def test_prompt_messages():
    chunks = [CHUNKS["ren1000-02"], CHUNKS["ficha-pedido-156-02"]]
    system, user = build_messages(" Quem conserta a lâmpada? ", chunks)
    assert system == {"role": "system", "content": SYSTEM_RAG}
    assert user["content"].startswith("Trechos:\n\n[T1] REN ANEEL 1.000/2021 > ")
    assert "\n\n[T2] " in user["content"] and user["content"].endswith("Pergunta: Quem conserta a lâmpada?")
    assert build_messages("Oi?", None)[0]["content"] == SYSTEM_NO_CONTEXT
    assert [chunk_label(i) for i in range(3)] == LABELS


def test_unsupported_data():
    support = "Emergência da Enel: 0800 72 72 196. O prazo é de trinta dias. E-mail 156@osasco.sp.gov.br."
    assert unsupported_data("Ligue 0800-727-2196.", support) == []
    assert unsupported_data("O prazo é de 30 dias.", support) == []
    assert unsupported_data("Ligue 0800 72 72 120.", support) == ["08007272120"]
    assert unsupported_data("Escreva para ouvidoria@osasco.sp.gov.br.", support) == ["ouvidoria@osasco.sp.gov.br"]


def test_unsupported_data_needs_the_whole_number():
    assert unsupported_data("A Central 156 recebeu 15 mil pedidos.", "A Central 156 recebeu 11.046 pedidos em 2023.") == ["15"]
    assert unsupported_data("O código tem 8 caracteres.", "WhatsApp (11) 3651-7080.") == ["8"]
    assert unsupported_data("Ligue para a Enel, 0800 72 72 196 (24 horas).", "Emergência: 0800 72 72 196, 24 horas.") == []
    assert unsupported_data("Foi a EC 132/2023.", "Emenda Constitucional nº 132, de 2023.") == []
    assert unsupported_data("Ligue (11) 3651-7080.", "WhatsApp 3651-7080.") == []
    assert unsupported_data("Ligue 3651-7080.", "WhatsApp (11) 3651-7080.") == []


@pytest.mark.parametrize("answer, missing", [
    ("O código tem oito caracteres.", ["8"]),  # por extenso também conta
    ("O código tem seis caracteres.", []),
    ("Ligue 0800 72 72 199.", ["08007272199"]),  # telefone remontado com pedaços de outros números
    ("Ligue 0800 72 72 1960.", ["080072721960"]),  # um dígito a mais não é o mesmo telefone
    ("Atende das 7:00 às 17:00.", []),  # hora cheia escrita de outro jeito
    ("Atende das 9:00 às 17:00.", ["9"]),
])
def test_unsupported_data_closes_the_gaps(answer, missing):
    support = "O código tem 6 caracteres. Emergência 0800 72 72 196; Defesa Civil 199. Atende das 7h às 17h."
    assert unsupported_data(answer, support) == missing


def record(answer=None, sources=("T1",), hits=("ren1000-02",), score=0.7, mode="rag", consta=True, usable=True):
    return {"mode": mode, "hits": [{"id": h, "score": score} for h in hits], "max_score": score if hits else None,
            "usable": usable, "model_consta": consta if answer is not None else False, "model_answer": answer or "",
            "model_sources": list(sources) if answer is not None else [], "refusal_reason": None}


def test_answer_classification():
    t01, t49 = CASES["t01"], CASES["t49"]
    good = record("A manutenção é do poder público municipal, a Prefeitura.")
    assert judge(t01, good, CHUNKS).outcome == CORRECT
    assert judge(t01, good, CHUNKS).cites_evidence is True
    assert judge(t01, good, CHUNKS, threshold=0.8).outcome == WRONG_REFUSAL
    assert judge(t01, record(None), CHUNKS).outcome == WRONG_REFUSAL
    assert judge(t01, record("A responsabilidade é da distribuidora."), CHUNKS).outcome == HALLUCINATION
    assert judge(t01, record("Ligue 0800 000 0000."), CHUNKS).outcome == HALLUCINATION
    assert judge(t01, record("Isso é tratado pela regulação."), CHUNKS).outcome == INSUFFICIENT
    assert judge(t01, record("Essa informação não consta nos trechos."), CHUNKS).outcome == DISGUISED_REFUSAL
    assert judge(t49, record(None, hits=("ficha-pedido-156-02",)), CHUNKS).outcome == CORRECT_REFUSAL
    assert judge(t49, record("Atende 24 horas.", hits=("ficha-pedido-156-02",)), CHUNKS).outcome == WRONG_ANSWER
    assert judge(t49, record("Não consta.", hits=("ficha-pedido-156-02",)), CHUNKS).outcome == DISGUISED_REFUSAL


@pytest.mark.parametrize("case_id, answer", [
    ("t10", "Não. O robô não mede em lux: o sensor LDR mede luminosidade relativa e não substitui o luxímetro."),
    ("t31", "Não use cabo de vassoura nem afaste o fio com nada: mantenha distância e ligue para a Emergência da Enel, 0800 72 72 196."),
    ("t30", "Não é com a Prefeitura, é com a Enel: ligue para a Emergência da Enel, 0800 72 72 196."),
    ("t02", "O de emergência é 0800 72 72 196, e o comercial é 0800 72 72 120."),
    ("t19", "Sim, pelo 156 ou pelo (11) 3651-7080; a Enel não conserta a iluminação pública."),
    ("t39", "A premissa está errada: a Enel é responsável pela rede elétrica, não pela iluminação pública, que é da Prefeitura. Peça pelo 156."),
    ("t15", "Na Ouvidoria Geral de Osasco, com o protocolo do 156. A ANEEL não recebe reclamações sobre iluminação pública."),
    ("t05", "Não está em vigor desde 2024: a edição de 2018 foi cancelada e substituída pela de 2024."),
    ("t36", "Em 2023 foram 11.046 pedidos. A outra parte da pergunta não consta na base de conhecimento."),
])
def test_correct_answers_with_negation_or_partial_marker(case_id, answer):
    record_ = record(answer, sources=(), hits=(), mode="no_context")
    assert judge(CASES[case_id], record_, CHUNKS).outcome == CORRECT


@pytest.mark.parametrize("case_id, answer", [
    ("t19", "Sim: ligue 156, não a Enel."),
    ("t19", "Sim, ligue no 156, e não na Enel."),
    ("t39", "A manutenção é da Prefeitura: ligue para o 156, e não para o 0800 72 72 196, quando a lâmpada apagar."),
    ("t10", "Não é verdade que o robô do Luz Segura mede em lux; o sensor mede luminosidade relativa."),
])
def test_corrections_and_negation_frames_are_not_hallucinations(case_id, answer):
    record_ = record(answer, sources=(), hits=(), mode="no_context")
    assert judge(CASES[case_id], record_, CHUNKS).outcome != HALLUCINATION


@pytest.mark.parametrize("case_id, answer", [
    ("t10", "Sim, o robô mede em lux, como um luxímetro."),
    ("t30", "É com a Prefeitura, que cuida da iluminação."),
    ("t02", "O telefone de emergência é 0800 72 72 120."),
    ("t30", "Não é com a Enel, é com a Prefeitura."),  # a negação é de outra oração
    ("t19", "Não ligue no 156, ligue para a Enel."),
    ("t01", "Não é a Prefeitura, a Enel é responsável pela manutenção."),
    ("t10", "O robô não é um luxímetro, mas mede a iluminância em lux."),
])
def test_wrong_affirmations_are_still_hallucinations(case_id, answer):
    record_ = record(answer, sources=(), hits=(), mode="no_context")
    assert judge(CASES[case_id], record_, CHUNKS).outcome == HALLUCINATION


def test_unanswerable_questions_with_content_are_wrong_answers():
    t49, t60 = CASES["t49"], CASES["t60"]
    hits = ("ficha-pedido-156-02",)
    partial = "O WhatsApp atende 24 horas. A outra parte da pergunta não consta na base de conhecimento."
    assert judge(t49, record(partial, hits=hits), CHUNKS).outcome == WRONG_ANSWER
    assert judge(t49, record("O horário do WhatsApp não consta na base.", hits=hits), CHUNKS).outcome == DISGUISED_REFUSAL
    promise = "Confirmo que o poste será consertado amanhã. Não tenho informação sobre o horário."
    assert judge(t60, record(promise, hits=hits), CHUNKS).outcome == WRONG_ANSWER
    full_refusal = "O horário do WhatsApp não consta na base; os demais canais também não constam com horário."
    assert judge(t49, record(full_refusal, hits=hits), CHUNKS).outcome == DISGUISED_REFUSAL


@pytest.mark.parametrize("mode", ["no_context", "lora_no_context"])
def test_modes_without_context_ignore_threshold(mode):
    t01 = CASES["t01"]
    answer = record("A manutenção é do poder público municipal, a Prefeitura.", sources=(), hits=(), mode=mode)
    judgement = judge(t01, answer, CHUNKS, threshold=0.42)
    assert judgement.outcome == CORRECT
    assert judgement.cites_evidence is None


def test_scoreboard_metrics():
    cases = [CASES["t01"], CASES["t49"]]
    summary = summarize(cases, [CORRECT, WRONG_ANSWER])
    assert summary["net_score"] == 0
    assert summary["answer_rate"] == (2, 2)
    assert summary["unsupported_among_answered"] == (1, 2)
    low, high = wilson(5, 10)
    assert 0.23 < low < 0.24 and 0.76 < high < 0.77
    test = mcnemar([CORRECT] * 10 + [WRONG_REFUSAL] * 2, [WRONG_REFUSAL] * 10 + [CORRECT] * 2)
    assert test["only_first_correct"] == 10 and test["only_second_correct"] == 2
    assert test["p_value"] == pytest.approx(0.0386, abs=1e-3)


def test_fingerprint_follows_the_questions_not_the_answer_key(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "CASES_DIR", tmp_path)
    case = CASES["t01"]
    (tmp_path / "a.jsonl").write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    changed_key = {**case, "reference_answer": "outra", "forbidden": ["x"], "why_absent": "y"}
    (tmp_path / "b.jsonl").write_bytes((json.dumps(changed_key, ensure_ascii=False) + "\r\n").encode("utf-8"))
    changed_question = {**case, "question": case["question"] + " Obrigado."}
    (tmp_path / "c.jsonl").write_text(json.dumps(changed_question, ensure_ascii=False) + "\n", encoding="utf-8")
    server = {"model": "modelo.gguf", "build": "b1"}
    assert configuration_fingerprint("rag", "a.jsonl", server) == configuration_fingerprint("rag", "b.jsonl", server)
    assert configuration_fingerprint("rag", "a.jsonl", server) != configuration_fingerprint("rag", "c.jsonl", server)
    assert configuration_fingerprint("rag", "a.jsonl", server) != configuration_fingerprint("no_context", "a.jsonl", server)
    with pytest.raises(ValueError):
        configuration_fingerprint("rag_lora", "a.jsonl", server)
    adapter = tmp_path / "epoca_2.gguf"
    adapter.write_bytes(b"adaptador de teste")
    with pytest.raises(ValueError):  # o servidor não carregou esse adaptador
        configuration_fingerprint("rag_lora", "a.jsonl", {**server, "adapters": ["epoca_1.gguf"]}, adapter)
    loaded = {**server, "adapters": ["epoca_2.gguf"]}
    assert configuration_fingerprint("rag_lora", "a.jsonl", loaded, adapter) != configuration_fingerprint("rag", "a.jsonl", loaded)
