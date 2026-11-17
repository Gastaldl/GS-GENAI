"""Avaliação: recall da busca, classificação de cada resposta e métricas do placar.

Os pontos-chave e os padrões proibidos de cada caso são expressões regulares aplicadas
ao texto em minúsculas e sem acentos. Padrões que começam com "#" são comparados só com
os dígitos do texto (telefones podem vir com ou sem espaços e hífens). Um padrão proibido
só conta quando aparece afirmado: logo depois de "não", "nunca", "nem" ou "jamais" na mesma
oração ("não mede em lux", "nunca suba no poste") ou dentro de uma correção ("ligue 156, não
a Enel"), a ocorrência não conta (veja affirmed).

Desfechos de cada caso:
- CORRETA: respondeu e cobriu todos os pontos-chave, sem dado proibido nem dado sem apoio;
- INSUFICIENTE: respondeu sem erro detectado, mas faltou ponto-chave;
- ALUCINACAO: respondeu com padrão proibido ou com número, e-mail ou site que não está
  no material de apoio (trechos entregues ao modelo; sem contexto, o gabarito);
- RECUSA_DISFARCADA: marcou consta true, mas o texto só diz que a informação não consta;
- RECUSA_INDEVIDA: recusou uma pergunta que a base responde;
- RECUSA_CORRETA: recusou uma pergunta que a base não responde;
- RESPOSTA_INDEVIDA: respondeu uma pergunta que a base não responde. Isso inclui o texto que
  mistura recusa com conteúdo: padrão proibido, dado sem apoio ou a marca de resposta parcial
  ("a outra parte da pergunta não consta"), que afirma ter respondido uma parte.
Placar líquido por caso: +1 para CORRETA e RECUSA_CORRETA, -1 para ALUCINACAO e
RESPOSTA_INDEVIDA, 0 para os demais. "Resposta sem apoio entre respondidas" é
(ALUCINACAO + RESPOSTA_INDEVIDA) dividido pelas respostas dadas (casos não recusados).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from luz_segura.knowledge_base import ROOT, Chunk

CASES_DIR = ROOT / "avaliacao"
RUNS_DIR = CASES_DIR / "saidas"

# Desfechos possíveis de cada caso
CORRECT = "CORRETA"
INSUFFICIENT = "INSUFICIENTE"
HALLUCINATION = "ALUCINACAO"
DISGUISED_REFUSAL = "RECUSA_DISFARCADA"
WRONG_REFUSAL = "RECUSA_INDEVIDA"
CORRECT_REFUSAL = "RECUSA_CORRETA"
WRONG_ANSWER = "RESPOSTA_INDEVIDA"
OUTCOMES = [CORRECT, INSUFFICIENT, HALLUCINATION, DISGUISED_REFUSAL, WRONG_REFUSAL, CORRECT_REFUSAL, WRONG_ANSWER]
NET_SCORE = {CORRECT: 1, CORRECT_REFUSAL: 1, INSUFFICIENT: 0, DISGUISED_REFUSAL: 0, WRONG_REFUSAL: 0,
             HALLUCINATION: -1, WRONG_ANSWER: -1}
SUCCESS = {CORRECT, CORRECT_REFUSAL}
REFUSED = {WRONG_REFUSAL, CORRECT_REFUSAL}


def load_cases(name: str) -> list[dict]:
    with open(CASES_DIR / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents).strip()


def only_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def matches(pattern: str, text: str) -> bool:
    if pattern.startswith("#"):
        return pattern[1:] in only_digits(text)
    return re.search(pattern, normalize_text(text)) is not None


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança de Wilson (95%) para uma proporção."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, center - half), min(1.0, center + half))


def format_rate(successes: int, n: int) -> str:
    low, high = wilson(successes, n)
    return f"{successes}/{n} = {100 * successes / n:.1f}% ({100 * low:.1f} a {100 * high:.1f})" if n else "n/a"


# ---------------------------------------------------------------- busca

@dataclass
class RetrievalResult:
    case_id: str
    top_ids: list
    top_scores: list
    groups_hit_at: dict  # k -> número de grupos de evidência cobertos no top-k
    n_groups: int


def retrieval_results(cases: list[dict], score_matrix: np.ndarray, chunk_ids: list[str], max_k: int = 5) -> list[RetrievalResult]:
    results = []
    for case, row in zip(cases, score_matrix):
        order = np.argsort(-row)[:max_k]
        top_ids = [chunk_ids[i] for i in order]
        groups = case["gold_chunks"]
        hit_at = {k: sum(any(c in top_ids[:k] for c in group) for group in groups) for k in range(1, max_k + 1)}
        results.append(RetrievalResult(case["id"], top_ids, [float(row[i]) for i in order], hit_at, len(groups)))
    return results


def recall_at(results: list[RetrievalResult], k: int) -> float:
    """Média, por pergunta, da fração de grupos de evidência com algum trecho no top-k."""
    values = [r.groups_hit_at[k] / r.n_groups for r in results if r.n_groups]
    return float(np.mean(values)) if values else math.nan


def full_coverage_at(results: list[RetrievalResult], k: int) -> tuple[int, int]:
    """Quantas perguntas têm todos os grupos de evidência no top-k (teto do acerto do RAG)."""
    valid = [r for r in results if r.n_groups]
    return sum(r.groups_hit_at[k] == r.n_groups for r in valid), len(valid)


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- dados sem apoio

NUMBER_WORDS = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6, "sete": 7,
    "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12, "treze": 13, "quatorze": 14, "catorze": 14,
    "quinze": 15, "dezesseis": 16, "dezessete": 17, "dezoito": 18, "dezenove": 19, "vinte": 20,
    "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60, "setenta": 70, "oitenta": 80,
    "noventa": 90, "cem": 100, "mil": 1000,
}
TENS = r"(vinte|trinta|quarenta|cinquenta|sessenta|setenta|oitenta|noventa)"
UNITS = r"(um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito|nove)"
# Grupo numérico: dígitos ligados por espaço, hífen, ponto, barra ou vírgula; parênteses só
# em volta de um DDD ("(11) 3651-7080", "15/11/2026", "4,4642" e "0800 72 72 196" são um grupo
# cada; em "0800 72 72 196 (24 horas)" o "24" é outro grupo).
NUMBER_GROUP = re.compile(r"\(?\d(?:\d|[ \-./,](?=\d)|\s?\((?=\d{2}\))|\)\s?(?=\d))*")
NUMBER_PARTS = re.compile(r"[ \-./,()]+")
WHOLE_HOUR = re.compile(r"\b0?(\d{1,2})(?::|h)00\b")  # "8:00", "08h00" e "8h" são a mesma hora
PHONE_LENGTH = 8  # telefones: o mesmo número com DDD (2 dígitos) ou com 55 + DDD (4 dígitos) também vale
ANSWER_NUMBER_WORDS = {w: v for w, v in NUMBER_WORDS.items() if w not in ("um", "uma", "mil")}
ADDRESS = re.compile(r"[a-z0-9._+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+|\b(?:[a-z0-9-]+\.)+(?:br|com|org|net|gov|app)\b")


def number_groups(text: str) -> set[str]:
    return {only_digits(g) for g in NUMBER_GROUP.findall(WHOLE_HOUR.sub(r"\1h", text))}


def number_parts(group: str) -> set[str]:
    """Partes de um grupo numérico separadas por espaço, hífen, ponto, barra, vírgula ou parênteses."""
    return {p for p in NUMBER_PARTS.split(group) if p}


def written_numbers(text: str, words: dict[str, int]) -> set[str]:
    """Números escritos por extenso ("seis", "trinta e cinco"), como dígitos."""
    normalized = normalize_text(text)
    found = {str(value) for word, value in words.items() if re.search(rf"\b{word}\b", normalized)}
    for tens, units in re.findall(rf"\b{TENS} e {UNITS}\b", normalized):
        found.add(str(NUMBER_WORDS[tens] + NUMBER_WORDS[units]))
    return found


def support_numbers(text: str) -> tuple[set[str], set[str]]:
    """Números do material de apoio: os grupos inteiros (só dígitos) e tudo o que conta como
    apoio (grupos inteiros, partes de grupos e números escritos por extenso)."""
    wholes, known = set(), set()
    for group in NUMBER_GROUP.findall(WHOLE_HOUR.sub(r"\1h", text)):
        wholes.add(only_digits(group))
        known |= number_parts(group)
    wholes |= written_numbers(text, NUMBER_WORDS)
    return wholes, known | wholes


def same_phone(digits: str, other: str) -> bool:
    """O mesmo telefone com e sem DDD (ou com 55 + DDD): o maior termina no menor e tem 2 ou 4 dígitos a mais."""
    short, long = sorted((digits, other), key=len)
    return len(short) >= PHONE_LENGTH and len(long) - len(short) in (2, 4) and long.endswith(short)


def supported_number(group: str, wholes: set[str], known: set[str]) -> bool:
    """Um número da resposta tem apoio se for igual a um grupo ou a uma parte de grupo do apoio, se
    for o mesmo telefone com ou sem DDD, ou, num grupo com barra ("132/2023", "13.460/2017"), se
    todas as partes tiverem apoio ("nº 132, de 2023")."""
    digits = only_digits(group)
    if digits in known or any(same_phone(digits, w) for w in wholes):
        return True
    return "/" in group and all(p in known for p in number_parts(group))


def unsupported_data(answer: str, support: str) -> list[str]:
    """Números (em dígitos ou por extenso), e-mails e sites da resposta que não aparecem no material de apoio."""
    wholes, known = support_numbers(support)
    groups = NUMBER_GROUP.findall(WHOLE_HOUR.sub(r"\1h", answer))
    missing = {only_digits(g) for g in groups if not supported_number(g, wholes, known)}
    missing |= {n for n in written_numbers(answer, ANSWER_NUMBER_WORDS) if n not in known}
    normalized_support = normalize_text(support)
    return sorted(missing) + [a for a in sorted(set(ADDRESS.findall(normalize_text(answer)))) if a not in normalized_support]


# ---------------------------------------------------------------- classificação das respostas

REFUSAL_LANGUAGE = [
    r"\bnao consta",
    r"\bnao (ha|tem|traz|trazem|informa|informam|apresenta|apresentam|menciona|mencionam|existe|existem|possuo|tenho)\b"
    r"[^.]{0,40}\b(informac|dado|registro|resposta)",
    r"\b(os )?trechos (fornecidos )?nao\b",
    r"\bnao (e|eh) possivel (responder|informar|saber)",
    r"\bnao sei\b",
]


def is_refusal_language(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.search(p, normalized) for p in REFUSAL_LANGUAGE)


# Marca de resposta parcial: diz que só "a outra parte" não consta, logo afirma ter respondido o
# resto. "O restante também não consta" é recusa completa e não conta.
PARTIAL_ANSWER = r"\b(outra parte|restante|o resto|segunda parte|demais)\b(?:(?!\btambem\b)[^.]){0,60}\bnao consta"
NEGATION = re.compile(r"\b(nao|nunca|nem|jamais)\b")
NEGATION_FRAME = re.compile(r"\bnao (e|eh) verdade que\b|\b(e|eh) falso que\b|\bem hipotese alguma\b")
CLAUSE_BREAK = re.compile(r"[.!?;:,]|\b(?:mas|porem|contudo|entretanto|no entanto|e sim)\b")
CORRECTION_INSIDE = re.compile(r",\s*(e\s+)?(nao|nunca|jamais)\b")
NEGATION_WINDOW = 25  # caracteres antes da ocorrência, na mesma oração


def affirmed(pattern: str, text: str) -> bool:
    """True se o padrão proibido aparece afirmado no texto. Não contam: a ocorrência logo depois
    de uma negação na mesma oração ("não mede em lux", "nunca afaste o fio com...", "não é verdade
    que..."), e a ocorrência que atravessa uma correção ("ligue 156, não a Enel"). A negação de
    outra oração não vale: em "não é com a Enel, é com a Prefeitura", a segunda parte é afirmada."""
    if pattern.startswith("#"):
        return matches(pattern, text)
    normalized = normalize_text(text)
    for found in re.finditer(pattern, normalized):
        if CORRECTION_INSIDE.search(found.group()):
            continue
        clause = CLAUSE_BREAK.split(normalized[:found.start()])[-1]
        if NEGATION.search(clause[-NEGATION_WINDOW:]) or NEGATION_FRAME.search(clause):
            continue
        return True
    return False


def uses_context(record: dict) -> bool:
    """True se a execução passou trechos ao modelo (RAG); False nos modos sem contexto."""
    from luz_segura.assistant import MODES

    return MODES[record["mode"]]["context"]


def support_text(case: dict, record: dict, chunks: dict[str, Chunk]) -> str:
    """Material de apoio da resposta: os trechos entregues ao modelo (RAG) ou, nos
    modos sem contexto, o gabarito e os trechos de evidência do caso."""
    parts = [case["question"]]
    if not uses_context(record):
        parts.append(case["reference_answer"])
        parts += [chunks[c].indexed_text for group in case["gold_chunks"] for c in group]
    else:
        parts += [chunks[h["id"]].indexed_text for h in record["hits"]]
    return "\n".join(parts)


@dataclass
class Judgement:
    case_id: str
    outcome: str
    refusal_reason: str | None
    key_points_hit: list
    forbidden_hit: list
    unsupported: list
    cites_evidence: bool | None  # só para respostas a perguntas respondíveis no RAG


def decision(record: dict, threshold: float | None) -> str | None:
    """Motivo de recusa do sistema completo, ou None se ele responde. As saídas são geradas
    com limiar 0 e o limiar é aplicado aqui, para que a mesma execução sirva a qualquer limiar."""
    if uses_context(record) and threshold is not None and record["max_score"] < threshold:
        return "limiar"
    if not record["usable"]:
        return "validacao"
    if not record["model_consta"]:
        return "modelo"
    return None


def cited_ids(record: dict) -> list[str]:
    by_label = {f"T{i + 1}": h["id"] for i, h in enumerate(record["hits"])}
    return [by_label[label] for label in record["model_sources"] or [] if label in by_label]


def judge(case: dict, record: dict, chunks: dict[str, Chunk], threshold: float | None = None) -> Judgement:
    expected_answer = case["expected"] == "answer"
    reason = decision(record, threshold)
    if reason is not None:
        outcome = WRONG_REFUSAL if expected_answer else CORRECT_REFUSAL
        return Judgement(case["id"], outcome, reason, [], [], [], None)
    text = record["model_answer"] or ""
    key_hits = [any(matches(p, text) for p in group) for group in case["key_points"]]
    forbidden = [p for p in case["forbidden"] if affirmed(p, text)]
    unsupported = unsupported_data(text, support_text(case, record, chunks))
    cites = None
    if expected_answer and uses_context(record):
        gold = {c for group in case["gold_chunks"] for c in group}
        cites = bool(gold & set(cited_ids(record)))
    if not expected_answer:
        mixed = forbidden or unsupported or matches(PARTIAL_ANSWER, text)
        outcome = DISGUISED_REFUSAL if is_refusal_language(text) and not mixed else WRONG_ANSWER
    elif forbidden or unsupported:
        outcome = HALLUCINATION
    elif all(key_hits):
        outcome = CORRECT
    elif not any(key_hits) and is_refusal_language(text):
        outcome = DISGUISED_REFUSAL
    else:
        outcome = INSUFFICIENT
    return Judgement(case["id"], outcome, None, key_hits, forbidden, unsupported, cites)


def always_refuse(cases: list[dict]) -> list[str]:
    """Linha de base: o sistema que recusa tudo."""
    return [WRONG_REFUSAL if c["expected"] == "answer" else CORRECT_REFUSAL for c in cases]


# ---------------------------------------------------------------- métricas do placar

def summarize(cases: list[dict], outcomes: list[str], records: list[dict] | None = None,
              judgements: list[Judgement] | None = None) -> dict:
    n = len(cases)
    counts = Counter(outcomes)
    answerable = [o for c, o in zip(cases, outcomes) if c["expected"] == "answer"]
    unanswerable = [o for c, o in zip(cases, outcomes) if c["expected"] == "refuse"]
    answered = [o for o in outcomes if o not in REFUSED]
    unsupported = counts[HALLUCINATION] + counts[WRONG_ANSWER]
    summary = {
        "cases": n,
        "outcomes": {o: counts[o] for o in OUTCOMES},
        "net_score": sum(NET_SCORE[o] for o in outcomes) / n,
        "accuracy": (counts[CORRECT] + counts[CORRECT_REFUSAL], n),
        "correct_among_answerable": (answerable.count(CORRECT), len(answerable)),
        "correct_refusals_among_unanswerable": (unanswerable.count(CORRECT_REFUSAL), len(unanswerable)),
        "answer_rate": (len(answered), n),
        "unsupported_among_answered": (unsupported, len(answered)),
    }
    if records is not None:
        generated = [r for r in records if r["raw_output"] is not None]
        summary["json_strict"] = (sum(bool(r["json_strict"]) for r in generated), len(generated))
        summary["json_usable"] = (sum(bool(r["usable"]) for r in generated), len(generated))
        summary["seconds_per_question"] = float(np.mean([r["seconds"] for r in records]))
    if judgements is not None:
        with_citation = [j for j in judgements if j.cites_evidence is not None]
        if with_citation:
            summary["cites_evidence"] = (sum(j.cites_evidence for j in with_citation), len(with_citation))
    return summary


SCOREBOARD_LABELS = {
    "accuracy": "acerto", "correct_among_answerable": "corretas (respondíveis)",
    "correct_refusals_among_unanswerable": "recusas corretas (sem resposta)", "answer_rate": "taxa de resposta",
    "unsupported_among_answered": "resposta sem apoio entre respondidas", "cites_evidence": "cita trecho de evidência",
    "json_strict": "JSON estrito",
}
MCNEMAR_LABELS = {"only_first_correct": "só a primeira acerta", "only_second_correct": "só a segunda acerta",
                  "p_value": "p-valor"}


def scoreboard_row(label: str, summary: dict) -> dict:
    """Linha do placar em texto, com intervalo de Wilson (95%) nas proporções."""
    row = {"configuração": label, "placar líquido": round(summary["net_score"], 3)}
    for key, name in SCOREBOARD_LABELS.items():
        if key in summary:
            row[name] = format_rate(*summary[key])
    if "seconds_per_question" in summary:
        row["s por pergunta"] = round(summary["seconds_per_question"], 2)
    return row


def markdown_table(frame) -> str:
    """Tabela Markdown de um DataFrame, sem a dependência extra que o pandas pede em to_markdown."""
    header = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, row in frame.iterrows():
        lines.append("| " + " | ".join([str(index)] + ["" if pd_isna(v) else str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def pd_isna(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def mcnemar(first: list[str], second: list[str]) -> dict:
    """Teste de McNemar exato (binomial bilateral) sobre acerto pareado por caso."""
    only_first = sum(a in SUCCESS and b not in SUCCESS for a, b in zip(first, second))
    only_second = sum(b in SUCCESS and a not in SUCCESS for a, b in zip(first, second))
    total = only_first + only_second
    tail = sum(math.comb(total, i) for i in range(min(only_first, only_second) + 1)) / 2 ** total if total else 1.0
    return {"only_first_correct": only_first, "only_second_correct": only_second, "p_value": min(1.0, 2 * tail)}


# ---------------------------------------------------------------- execuções salvas

RUN_TAGS = {"no_context": "sem_contexto", "rag": "rag", "rag_lora": "rag_lora", "lora_no_context": "lora_sem_contexto"}


def run_file(split: str, mode: str, suffix: str = "") -> Path:
    """Arquivo das saídas brutas de uma configuração numa partição (ex.: saidas/teste_sem_contexto.jsonl)."""
    return RUNS_DIR / f"{split}_{RUN_TAGS[mode]}{suffix}.jsonl"


def run_fingerprint(**parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def content_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


DENSE_RETRIEVAL = "BAAI/bge-m3 k=3"  # a busca corrigida usa HybridRetriever.label


def configuration_fingerprint(mode: str, cases_name: str, server_description: dict,
                              adapter: Path | None = None, retrieval: str = DENSE_RETRIEVAL) -> str:
    """Tudo o que muda a saída do modelo numa execução: configuração, modelo, versão do llama.cpp,
    parâmetros de geração, formato das mensagens (prompts de sistema e mensagem do usuário),
    texto dos trechos, perguntas dos casos, busca e, com o LoRA ligado, o adaptador.

    O gabarito dos casos (resposta de referência, pontos-chave, padrões proibidos, justificativas)
    fica de fora, porque não muda a saída do modelo: a avaliação o lê depois. A base e as
    perguntas entram pelo conteúdo, sem depender do fim de linha dos arquivos."""
    from luz_segura.assistant import MODES
    from luz_segura.knowledge_base import CHUNKS_FILE, load_chunks
    from luz_segura.llm import GENERATION, file_id
    from luz_segura.prompt import build_messages

    if MODES[mode]["lora_scale"]:
        if adapter is None:
            raise ValueError("configuração com o LoRA ligado: informe o arquivo do adaptador")
        loaded = server_description.get("adapters", [])
        if loaded != [Path(adapter).name]:
            raise ValueError(f"o llama-server carregou {loaded}, não {Path(adapter).name}; reinicie-o com esse adaptador")
    messages = [build_messages("PERGUNTA", None), build_messages("PERGUNTA", load_chunks()[:1])]
    questions = [[c["id"], c["question"]] for c in load_cases(cases_name)]
    return run_fingerprint(
        mode=mode, model=server_description["model"], build=server_description["build"], generation=GENERATION,
        messages=content_digest(messages), questions=content_digest(questions), retrieval=retrieval,
        chunks=content_digest(CHUNKS_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")),
        adapter=file_id(adapter) if MODES[mode]["lora_scale"] else None,
    )


def run_cases(assistant, cases: list[dict], mode: str, path: Path, fingerprint: str,
              progress: bool = True) -> list[dict]:
    """Roda os casos numa configuração e salva as saídas brutas. O assistente deve ter
    limiar 0: o limiar entra depois, em decision(). Se o arquivo já existe com a mesma
    impressão digital (prompt, modelo, adaptador, base, casos), reaproveita as saídas."""
    from luz_segura.assistant import result_record

    if path.exists():
        rows = read_jsonl(path)
        if rows and rows[0].get("fingerprint") == fingerprint and len(rows) == len(cases) + 1:
            return rows[1:]
    records = []
    for i, case in enumerate(cases, 1):
        result = assistant.run(case["question"], mode, generate_below_threshold=True)
        records.append({"case_id": case["id"], **result_record(result)})
        if progress and i % 10 == 0:
            print(f"  {mode}: {i}/{len(cases)}", flush=True)
    save_jsonl([{"fingerprint": fingerprint, "mode": mode, "cases": len(cases)}] + records, path)
    return records
