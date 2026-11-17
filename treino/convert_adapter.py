"""Converte o adaptador LoRA do formato PEFT (saída do notebook 03) para GGUF, o formato do llama.cpp.

Uso:
    python treino/convert_adapter.py <caminho/do/adaptador_luz_segura.zip>
    python treino/convert_adapter.py <pasta_peft> <saida.gguf>

Com o zip, o script extrai o conteúdo numa pasta nova (modelos/treino_lora/), converte cada
época (epoch_N/) para modelos/epocas/epoca_N.gguf e copia as métricas do treino e o exemplo
formatado para modelos/adaptador/ (metricas_treino.json e exemplo_formatado.txt). A escolha
da época fica para o notebook 04. O zip tem cerca de 200 MB: deixe-o fora do repositório.

A conversão usa o script convert_lora_to_gguf.py do código-fonte do llama.cpp b12359, num
ambiente Python com requirements/requirements-convert_lora_to_gguf.txt (veja o README).
Variáveis de ambiente:
- LUZ_LLAMA_SOURCE: pasta do código-fonte do llama.cpp (padrão: ferramentas/llama-b12359-fonte);
- LUZ_CONVERSION_PYTHON: executável Python desse ambiente (padrão: ferramentas/venv-conversao/
  Scripts/python.exe no Windows e ferramentas/venv-conversao/bin/python nos outros sistemas).
A configuração do modelo base (config.json do Qwen3-4B-Instruct-2507) fica em modelos/base/hf-config.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
LLAMA_SOURCE = Path(os.environ.get("LUZ_LLAMA_SOURCE", ROOT / "ferramentas" / "llama-b12359-fonte"))
CONVERSION_PYTHON = Path(os.environ.get("LUZ_CONVERSION_PYTHON", ROOT / "ferramentas" / "venv-conversao" / Path(*DEFAULT_PYTHON)))
BASE_CONFIG = ROOT / "modelos" / "base" / "hf-config"
EXTRACT_DIR = ROOT / "modelos" / "treino_lora"
EPOCHS_DIR = ROOT / "modelos" / "epocas"
ADAPTER_DIR = ROOT / "modelos" / "adaptador"
TRAINING_FILES = {"training_metrics.json": "metricas_treino.json", "rendered_example.txt": "exemplo_formatado.txt"}


def convert(peft_dir: Path, output: Path) -> None:
    # caminhos absolutos: a conferência abaixo e o conversor (que roda com cwd=ROOT) veem os mesmos arquivos
    peft_dir, output = Path(peft_dir).resolve(), Path(output).resolve()
    if not (peft_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"{peft_dir} não tem adapter_config.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(CONVERSION_PYTHON), str(LLAMA_SOURCE / "convert_lora_to_gguf.py"), "--base", str(BASE_CONFIG),
               "--outtype", "f16", "--outfile", str(output), str(peft_dir)]
    subprocess.run(command, check=True, cwd=ROOT)


def epoch_folders(folder: Path) -> list[tuple[int, Path]]:
    found = [(int(m.group(1)), p) for p in folder.iterdir() if p.is_dir() and (m := re.fullmatch(r"epoch_(\d+)", p.name))]
    return sorted(found)


def convert_zip(zip_path: Path) -> list[Path]:
    """Extrai o zip do notebook 03, converte todas as épocas e copia as métricas do treino."""
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    EXTRACT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(EXTRACT_DIR)
    epochs = epoch_folders(EXTRACT_DIR)
    if not epochs:
        raise FileNotFoundError(f"{zip_path} não tem pastas epoch_N")
    shutil.rmtree(EPOCHS_DIR, ignore_errors=True)
    outputs = []
    for epoch, folder in epochs:
        output = EPOCHS_DIR / f"epoca_{epoch}.gguf"
        convert(folder, output)
        outputs.append(output)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    for source, target in TRAINING_FILES.items():
        shutil.copyfile(EXTRACT_DIR / source, ADAPTER_DIR / target)
    return outputs


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".zip"):
        for path in convert_zip(Path(sys.argv[1])):
            print(path.relative_to(ROOT), path.stat().st_size, "bytes")
    elif len(sys.argv) == 3:
        convert(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
