"""Base de conhecimento: lê os documentos de base/documentos e gera os trechos.

Cada documento é um Markdown com um cabeçalho (chave: valor) entre linhas "---".
Cada seção "## título" vira um trecho. Um comentário "<!-- url: ... -->" logo abaixo
do título indica a fonte específica daquela seção; sem ele vale a URL do documento.

Uso: python -m luz_segura.knowledge_base
(gera base/trechos.jsonl, base/fontes.csv e base/fontes.md)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = ROOT / "base" / "documentos"
CHUNKS_FILE = ROOT / "base" / "trechos.jsonl"
MAX_WORDS = 250
REQUIRED_KEYS = ("id", "title", "short_title", "publisher", "url", "original_format", "kind", "license", "accessed")


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    document_title: str
    section: str
    text: str
    url: str
    kind: str
    license: str
    accessed: str

    @property
    def indexed_text(self) -> str:
        """Texto usado na busca: documento e seção entram junto com o conteúdo."""
        return f"{self.document_title} > {self.section}\n{self.text}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Document:
    id: str
    header: dict
    sections: list  # lista de (título, url, texto)


def read_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    parts = raw.split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        raise ValueError(f"{path.name}: cabeçalho ausente")
    header = {}
    for line in parts[1].strip().splitlines():
        key, _, value = line.partition(":")
        header[key.strip()] = value.strip()
    for key in REQUIRED_KEYS:
        if not header.get(key):
            raise ValueError(f"{path.name}: falta '{key}' no cabeçalho")
    if header["id"] != path.stem:
        raise ValueError(f"{path.name}: id '{header['id']}' difere do nome do arquivo")

    sections = []
    for block in re.split(r"^## ", parts[2], flags=re.MULTILINE)[1:]:
        title, _, body = block.partition("\n")
        url = header["url"]
        marker = re.match(r"\s*<!--\s*url:\s*(\S+)\s*-->", body)
        if marker:
            url = marker.group(1)
            body = body[marker.end():]
        text = body.strip()
        if not text:
            raise ValueError(f"{path.name}: seção vazia '{title}'")
        sections.append((title.strip(), url, text))
    if not sections:
        raise ValueError(f"{path.name}: nenhuma seção '## '")
    return Document(header["id"], header, sections)


def read_documents(folder: Path = DOCUMENTS_DIR) -> list[Document]:
    return [read_document(path) for path in sorted(folder.glob("*.md"))]


def build_chunks(documents: list[Document]) -> list[Chunk]:
    chunks = []
    for doc in documents:
        h = doc.header
        for number, (section, url, text) in enumerate(doc.sections, start=1):
            if len(text.split()) > MAX_WORDS:
                raise ValueError(f"{doc.id}: seção '{section}' passa de {MAX_WORDS} palavras")
            chunks.append(Chunk(
                id=f"{doc.id}-{number:02d}", document=doc.id, document_title=h["short_title"],
                section=section, text=text, url=url, kind=h["kind"], license=h["license"], accessed=h["accessed"],
            ))
    duplicated = [i for i, n in Counter(c.id for c in chunks).items() if n > 1]
    if duplicated:
        raise ValueError(f"ids repetidos: {duplicated}")
    return chunks


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_FILE) -> str:
    """Grava os trechos e devolve o sha256 dos bytes gravados (o mesmo que os manifestos registram)."""
    content = "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks(path: Path = CHUNKS_FILE) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def sources_table(documents: list[Document], chunks: list[Chunk]) -> list[dict]:
    """Uma linha por documento. section_urls traz os endereços próprios de seções (marcados com
    <!-- url: ... --> no documento) e other_urls, as outras fontes consultadas que o cabeçalho
    lista (chave opcional other_urls, separada por espaços)."""
    per_document = Counter(c.document for c in chunks)
    section_urls: dict[str, list[str]] = {}
    for chunk in chunks:
        urls = section_urls.setdefault(chunk.document, [])
        if chunk.url not in urls:
            urls.append(chunk.url)
    rows = []
    for doc in documents:
        main_url = doc.header["url"]
        rows.append({
            "document": doc.id, "source": doc.header["title"], "publisher": doc.header["publisher"],
            "url": main_url, "original_format": doc.header["original_format"], "kind": doc.header["kind"],
            "license": doc.header["license"], "accessed": doc.header["accessed"], "chunks": per_document[doc.id],
            "section_urls": " ".join(u for u in section_urls.get(doc.id, []) if u != main_url),
            "other_urls": doc.header.get("other_urls", ""),
        })
    return rows


def link_from(folder: Path, url: str) -> str:
    """Endereço para usar num Markdown gravado em folder: caminhos do repositório viram relativos."""
    if url.startswith(("http://", "https://")):
        return url
    return Path(os.path.relpath(ROOT / url, folder)).as_posix()


def save_sources(rows: list[dict], folder: Path) -> None:
    with open(folder / "fontes.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = ("| Documento | Fonte | Formato original | Tipo e licença | Trechos | Outros endereços (seções e fontes consultadas) |\n"
              "|---|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        extra = (r["section_urls"].split() + r["other_urls"].split())
        others = " ".join(f"[{i}]({link_from(folder, u)})" for i, u in enumerate(extra, 1)) or "-"
        lines.append(f"| `{r['document']}` | [{r['source']}]({link_from(folder, r['url'])}) | {r['original_format']} | "
                     f"{r['kind']}. {r['license']} | {r['chunks']} | {others} |\n")
    total = sum(r["chunks"] for r in rows)
    (folder / "fontes.md").write_text(
        "# Fontes da base de conhecimento\n\nGerado por `python -m luz_segura.knowledge_base`. "
        f"Acesso de todas as fontes em 15/11/2026. {len(rows)} documentos, {total} trechos. A última coluna "
        "lista os endereços próprios de seções (cada trecho guarda o seu em base/trechos.jsonl) e as outras "
        "fontes consultadas para redigir as fichas.\n\n" + header + "".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    documents = read_documents()
    chunks = build_chunks(documents)
    digest = save_chunks(chunks)
    save_sources(sources_table(documents, chunks), CHUNKS_FILE.parent)
    sizes = sorted(c.word_count for c in chunks)
    print(f"{len(documents)} documentos, {len(chunks)} trechos, sha256 {digest[:16]}")
    print(f"palavras por trecho: mínimo {sizes[0]}, mediana {sizes[len(sizes) // 2]}, máximo {sizes[-1]}")
    for chunk in chunks:
        if chunk.word_count < 30:
            print(f"  aviso: {chunk.id} tem só {chunk.word_count} palavras")


if __name__ == "__main__":
    main()
