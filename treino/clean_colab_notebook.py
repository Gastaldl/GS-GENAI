"""Limpa os metadados de um notebook baixado do Colab antes de ele entrar no repositório.

Uso:
    python treino/clean_colab_notebook.py notebooks/03_treino_lora_colab.ipynb

O Colab grava em cada célula executada dados da conta Google (nome e identificador do usuário
em metadata.executionInfo) e outros identificadores internos. O script lê o .ipynb como JSON,
sem executar nada, e remove esses campos; as saídas das células ficam, porque são o registro do
treino. No fim, procura no texto gravado o que ainda pode identificar alguém: campos de conta do
Colab, endereços de e-mail (menos os públicos da base, de domínios .gov.br) e caminhos com pasta
de usuário (Users no Windows e no macOS, home no Linux). Se achar algum, avisa para conferir
antes do commit.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CELL_KEYS = ("executionInfo", "colab", "outputId", "id")
NOTEBOOK_COLAB_KEYS = ("provenance", "name", "include_colab_link", "authorship_tag")
ACCOUNT_FIELDS = r"displayName|userId|photoUrl|user_tz"
EMAIL = r"[\w.+-]+@(?![\w.-]*\.gov\.br\b)[\w-]+(?:\.[\w-]+)+"  # qualquer e-mail, menos os de .gov.br
# a verificação roda sobre o JSON gravado, em que cada barra invertida do caminho aparece dobrada
USER_FOLDER = r"[A-Za-z]:\\\\+Users\\\\+|/home/|/Users/"
SUSPICIOUS = re.compile("|".join((ACCOUNT_FIELDS, EMAIL, USER_FOLDER)))


def clean(notebook: dict) -> int:
    """Remove os campos de metadados do Colab e devolve quantos foram removidos."""
    removed = 0
    for cell in notebook.get("cells", []):
        metadata = cell.get("metadata", {})
        for key in CELL_KEYS:
            if key in metadata:
                del metadata[key]
                removed += 1
    colab = notebook.get("metadata", {}).get("colab", {})
    for key in NOTEBOOK_COLAB_KEYS:
        if key in colab:
            del colab[key]
            removed += 1
    return removed


def main(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    removed = clean(notebook)
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"{path}: {removed} campos de metadados removidos")
    leftovers = sorted(set(m.group(0) for m in SUSPICIOUS.finditer(text)))
    if leftovers:
        print("Atenção, confira antes do commit:", leftovers)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
