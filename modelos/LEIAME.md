# Modelos

## Modelo base (baixar)

| item | valor |
|---|---|
| modelo | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), Apache-2.0 |
| arquivo | `Qwen3-4B-Instruct-2507-Q8_0.gguf`, de [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) |
| tamanho | 4.280.405.600 bytes |
| sha256 | `391c1e410fd9f4cf2de2b510273b56a84c19ce18f4fa3bfb3774031dac4ef068` |
| commit no Hugging Face | `a06e946bb6b655725eafa393f4a9745d460374c9` (GGUF) e `cdbee75f17c01a7cc42f958dc650907174af0554` (configuração do modelo original) |

```
uv run hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q8_0.gguf --revision a06e946bb6b655725eafa393f4a9745d460374c9 --local-dir modelos/base
uv run hf download Qwen/Qwen3-4B-Instruct-2507 config.json tokenizer_config.json --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir modelos/base/hf-config
```

Conferência do arquivo: `Get-FileHash modelos/base/Qwen3-4B-Instruct-2507-Q8_0.gguf -Algorithm SHA256` (PowerShell) ou `sha256sum` (Linux).

A pasta `base/hf-config` só serve para converter o adaptador para GGUF.

## Servidor (llama.cpp)

Versão [b12359](https://github.com/ggml-org/llama.cpp/releases/tag/b12359). No Windows com GPU AMD ou NVIDIA, o pacote `llama-b12359-bin-win-vulkan-x64.zip` deve ser extraído em `ferramentas/llama-b12359-vulkan/`. Em outro sistema, use o binário da sua plataforma e indique o caminho na variável `LUZ_LLAMA_SERVER`.

Para converter adaptadores, use o código-fonte da mesma versão (`convert_lora_to_gguf.py`) num ambiente próprio:

```
git clone --depth 1 --branch b12359 https://github.com/ggml-org/llama.cpp ferramentas/llama-b12359-fonte
uv venv --python 3.12 ferramentas/venv-conversao
uv pip install --python ferramentas/venv-conversao -r ferramentas/llama-b12359-fonte/requirements/requirements-convert_lora_to_gguf.txt
```

Depois, `uv run python treino/convert_adapter.py <caminho do adaptador_luz_segura.zip>` converte todas as épocas (veja o passo 5 do README).

<!-- ADAPTADOR -->
