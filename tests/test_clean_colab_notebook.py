"""Limpeza do notebook baixado do Colab (treino/clean_colab_notebook.py), num notebook de exemplo."""
import json

import pytest

from treino.clean_colab_notebook import main

STREAM = {"name": "stdout", "output_type": "stream"}
WARNING = "Atenção, confira antes do commit"


def write_colab_notebook(folder, output_text):
    """Grava um notebook como o Colab baixa: conta Google e identificadores nos metadados e a saída
    do treino na célula."""
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {
                "id": "a1b2c3", "outputId": "d4e5f6", "colab": {"base_uri": "https://localhost:8080/"},
                "executionInfo": {"status": "ok", "user_tz": 180,
                                  "user": {"displayName": "Fulano de Tal", "userId": "0123456789"}},
            },
            "outputs": [{**STREAM, "text": [output_text]}],
            "source": ["treinar()"],
        }],
        "metadata": {"colab": {"provenance": [], "name": "03_treino_lora_colab.ipynb"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    path = folder / "03_treino_lora_colab.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    return path


def test_removes_colab_metadata_and_keeps_outputs(tmp_path, capsys):
    # o endereço público da base (domínio .gov.br) aparece nas saídas e não leva aviso
    output = "época 1: perda 0,42; exemplo cita 156@osasco.sp.gov.br\n"
    path = write_colab_notebook(tmp_path, output)
    main(path)
    cleaned = json.loads(path.read_text(encoding="utf-8"))
    cell = cleaned["cells"][0]
    assert cell["metadata"] == {}  # sem executionInfo, colab, outputId e id
    assert cell["outputs"] == [{**STREAM, "text": [output]}]
    assert cleaned["metadata"]["colab"] == {}
    assert cleaned["metadata"]["kernelspec"]["name"] == "python3"
    printed = capsys.readouterr().out
    assert "6 campos de metadados removidos" in printed
    assert WARNING not in printed


@pytest.mark.parametrize("leftover", [
    "aluno@fiap.com.br",
    "C:\\Users\\fulano\\Downloads\\adaptador_luz_segura.zip",
    "/home/fulano/adaptador_luz_segura.zip",
])
def test_warns_about_email_or_user_folder_left_in_outputs(tmp_path, capsys, leftover):
    main(write_colab_notebook(tmp_path, f"arquivo de {leftover}\n"))
    assert WARNING in capsys.readouterr().out
