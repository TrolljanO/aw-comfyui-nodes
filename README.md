# a|w ComfyUI Nodes

Custom nodes para o ComfyUI usado no Studio PPP (a|w Digital Lab).

## Nodes disponíveis

### GeminiImageInteractionsNode — `aw/gemini`

Chama a [Gemini Interactions API](https://ai.google.dev/gemini-api/docs/image-generation)
para gerar ou editar imagens em múltiplos turnos (`previous_interaction_id`).

#### Entradas

| Nome | Tipo | Obrigatório | Padrão | Descrição |
|---|---|---|---|---|
| `prompt` | STRING | sim | — | Instrução de geração/edição |
| `api_key` | STRING | sim | `GEMINI_API_KEY` | Nome da var de env ou literal |
| `model` | LIST | sim | `gemini-3-pro-image` | Modelo da API |
| `aspect_ratio` | LIST | sim | `16:9` | Proporção de saída |
| `image_size` | LIST | sim | `2K` | Resolução de saída |
| `mime_type` | LIST | sim | `image/png` | Formato de saída |
| `seed` | INT | sim | 0 | Ignorado pela API (compatibilidade) |
| `images` | IMAGE | não | — | Imagem(ns) de referência (tensor) |
| `previous_interaction_id` | STRING | não | `""` | ID do turno anterior |
| `system_instruction` | STRING | não | `""` | Instrução de sistema |
| `thinking_level` | LIST | não | `minimal` | Nível de raciocínio (só flash-3.1) |

#### Saídas

| Nome | Tipo | Descrição |
|---|---|---|
| `image` | IMAGE | Tensor `[1,H,W,3]` float32 |
| `interaction_id` | STRING | ID do turno para encadear em `previous_interaction_id` |

#### Exemplo de grafo (nodes do workflow c11)

```json
{
  "1": {"class_type": "LoadImage",   "inputs": {"image": "example.png"}},
  "6": {
    "class_type": "GeminiImageInteractionsNode",
    "inputs": {
      "prompt":                  "...",
      "api_key":                 "GEMINI_API_KEY",
      "model":                   "gemini-3-pro-image",
      "aspect_ratio":            "16:9",
      "image_size":              "2K",
      "mime_type":               "image/png",
      "seed":                    69,
      "images":                  ["1", 0],
      "previous_interaction_id": "",
      "system_instruction":      ""
    }
  },
  "7": {"class_type": "SaveImage",   "inputs": {"filename_prefix": "archstudio/c11", "images": ["6", 0]}},
  "9": {"class_type": "PreviewAny",  "inputs": {"source": ["6", 1]}}
}
```

`node "9"` (PreviewAny) expõe o `interaction_id` no `/history` do ComfyUI, permitindo
que o orquestrador encadeie o próximo turno sem re-enviar a imagem.

## Variável de ambiente

`GEMINI_API_KEY` deve existir no ambiente do processo ComfyUI.
Em desenvolvimento, crie um arquivo `.env` na raiz do pack (ignorado pelo git).

## Desenvolvimento local

```bash
uv run --with pytest --with requests --with python-dotenv \
       --with numpy --with pillow \
       python -m pytest -q
```
