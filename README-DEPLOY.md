# Deploy no ComfyUI (ECS/EFS) — v0.2.0

> **A tag `v0.2.0` so existe depois do merge desta entrega.** Enquanto a PR estiver
> aberta, `git checkout v0.2.0` falha com `pathspec did not match`. O passo de tag faz
> parte do release (ver "Release: criar a tag" no fim deste documento) e o deploy no
> ECS so deve acontecer depois dele. Para testar antes do merge, use a branch:
> `git checkout feat/aw-gpt-image-2.5-and-kling-video`.

Pack: `aw-comfyui-nodes` — tres nodes disponíveis a partir desta versao.

| Node | Classe | Endpoint/API |
|---|---|---|
| Gemini Image Interactions | `GeminiImageInteractionsNode` | Google Interactions API |
| GPT Image | `AwGptImageNode` | OpenAI `/images/edits` + `/images/generations` |
| Kling Video | `AwKlingVideoNode` | Kling AI text2video / image2video |

Requer **v0.2.0 ou maior**. A v0.1.0 carrega sem erro e **nao registra node nenhum**
(ver "Erros conhecidos"). A v0.1.1 registrava apenas GeminiImageInteractionsNode.

## Aviso de producao: homolog e prod compartilham o MESMO ComfyUI

O pool de ComfyUI roda na conta PROD (`912668123297`). A conta homolog (`828435023027`)
e exclusiva do Laravel. Qualquer node novo instalado no ComfyUI nasce VIVO em producao
no mesmo deploy — e seguro enquanto nenhum workflow habilitado o referenciar.

## Caminhos reais no host do ComfyUI

Confirme antes de qualquer comando:

```bash
ls -d /opt/content/custom_nodes
ls -d /opt/content/custom_nodes/ComfyUI-Manager
```

## Plano A — git clone direto (recomendado)

O ComfyUI carrega no boot qualquer pasta de `custom_nodes` que tenha `__init__.py`.

```bash
cd /opt/content/custom_nodes
git clone https://github.com/TrolljanO/aw-comfyui-nodes
cd aw-comfyui-nodes && git checkout v0.2.0

# dependencias (requests e python-dotenv normalmente ja existem no ComfyUI)
python -m pip install -r requirements.txt
```

Depois: rolling restart (secao abaixo).

## Plano B — ComfyUI-Manager

Pre-requisitos no `config.ini` do Manager:

```bash
find /opt/content -name config.ini -path '*Manager*'
```

```ini
[manager]
allow_git_url_install = true
security_level = normal-
```

- **Pela UI:** Manager → Install via Git URL → `https://github.com/TrolljanO/aw-comfyui-nodes` → Install
- **Pela CLI:**

```bash
python /opt/content/custom_nodes/ComfyUI-Manager/cm-cli.py \
    install https://github.com/TrolljanO/aw-comfyui-nodes
```

## Restart apos instalacao (ECS — DevOps executa)

```bash
aws ecs update-service \
    --cluster cluster-confyui-prd \
    --service svc-confyui-prd-sqs \
    --force-new-deployment \
    --profile <aws-profile-prod>
```

## Verificacao

### Prova que vale: GET /object_info/<NodeClass>

```bash
# Resposta com a classe = node registrado
# Resposta {} (objeto vazio) = pack presente na pasta mas node NAO registrado
curl -s http://<COMFYUI_HOST>/object_info/GeminiImageInteractionsNode \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys()))"
# Saida esperada: ['GeminiImageInteractionsNode']

curl -s http://<COMFYUI_HOST>/object_info/AwGptImageNode \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys()))"
# Saida esperada: ['AwGptImageNode']

curl -s http://<COMFYUI_HOST>/object_info/AwKlingVideoNode \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys()))"
# Saida esperada: ['AwKlingVideoNode']
```

**IMPORTANTE:** resposta `{}` significa que o pack esta na pasta mas o node nao foi
registrado. Isso ocorre quando o import e absoluto (`from nodes import ...`) em vez de
relativo (`from .nodes import ...`). Veja "Erros conhecidos" abaixo.

### Confirmar pack instalado (via Manager)

```bash
curl -s -u "$COMFY_USER:$COMFY_PASS" \
    "https://<COMFYUI_HOST>/v2/customnode/installed?mode=default" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); \
      print(d.get('aw-comfyui-nodes') or 'NAO ENCONTRADO')"
```

### Validar grafo sem custo (schema check)

```bash
# GeminiImageInteractionsNode
curl -s -X POST http://<COMFYUI_HOST>/prompt \
    -H 'Content-Type: application/json' \
    -d '{"prompt":{"1":{"class_type":"LoadImage","inputs":{"image":"__nonexistent__.png"}},"6":{"class_type":"GeminiImageInteractionsNode","inputs":{"prompt":"test","api_key":"GEMINI_API_KEY","model":"gemini-3-pro-image","aspect_ratio":"16:9","image_size":"2K","mime_type":"image/png","seed":0,"images":["1",0],"previous_interaction_id":"","system_instruction":""}}}}'

# AwGptImageNode
curl -s -X POST http://<COMFYUI_HOST>/prompt \
    -H 'Content-Type: application/json' \
    -d '{"prompt":{"6":{"class_type":"AwGptImageNode","inputs":{"prompt":"test","api_key":"OPENAI_API_KEY","model":"gpt-image-2","background":"auto","size":"1024x1024","quality":"auto","output_format":"png","output_compression":100,"n_images":1,"seed":0}}}}'

# AwKlingVideoNode
curl -s -X POST http://<COMFYUI_HOST>/prompt \
    -H 'Content-Type: application/json' \
    -d '{"prompt":{"6":{"class_type":"AwKlingVideoNode","inputs":{"prompt":"test","access_key":"KLING_ACCESS_KEY","secret_key":"KLING_SECRET_KEY","model_name":"kling-v3","negative_prompt":"","duration":"5","mode":"std","aspect_ratio":"16:9","sound":"off","filename_prefix":"aw_kling","timeout_seconds":600,"seed":0}}}}'
```

Se o node nao existe: response contem `"error"` com `"node_errors"`.
Se o node existe: response contem `"prompt_id"`.

## Variaveis de ambiente

| Variavel | Node | Descricao |
|---|---|---|
| `GEMINI_API_KEY` | GeminiImageInteractionsNode | API key do Google AI Studio |
| `OPENAI_API_KEY` | AwGptImageNode | API key da OpenAI |
| `KLING_ACCESS_KEY` | AwKlingVideoNode | Access Key ID do Kling AI |
| `KLING_SECRET_KEY` | AwKlingVideoNode | Secret Access Key do Kling AI |

Adicionar a task definition do servico ECS antes do rolling restart.

Os inputs de credencial aceitam tanto o nome da variavel de ambiente (ex: `OPENAI_API_KEY`)
quanto o valor literal da chave. Se o input for todo maiusculo e sem espacos, o node tenta
resolver via `os.environ` antes de usar como literal.

## Erros conhecidos

### `/object_info/<NodeClass>` devolve `{}` sem erro no log

Causa: import absoluto `from nodes import ...` em vez de relativo `from .nodes import ...`.
A raiz do ComfyUI, que esta no `sys.path`, tem um `nodes.py` proprio — o registro core.
O import absoluto resolve para o core, o pack carrega "com sucesso" e registra zero nodes.

Fix: verificar a versao instalada. A v0.1.0 tinha este bug; v0.1.1 corrigiu para
GeminiImageInteractionsNode; v0.2.0 estende o fix para os tres nodes.

```bash
cd /opt/content/custom_nodes/aw-comfyui-nodes
git fetch --tags && git checkout v0.2.0
python -m pip install -r requirements.txt
# + rolling restart
```

Este comportamento e coberto pelo teste automatico `tests/test_comfyui_entrypoint.py`.

### `SyntaxError: from __future__ import annotations must be at the beginning of the file`

Traceback termina em `ComfyUI-Manager/glob/node_package.py`. Nao e deste pack.
Enquanto o Manager estiver quebrado, use o Plano A (clone direto).

```bash
cd /opt/content/custom_nodes/ComfyUI-Manager
git checkout -- glob/node_package.py
```

### JWT expirado (Kling)

O JWT gerado pelo AwKlingVideoNode tem validade de 30 minutos (1800s). A geracao
ocorre a cada chamada; nao ha cache. Se o host do ComfyUI tiver relogio muito
defasado (NTP desativado), o `nbf` (not-before = now-5s) pode ser rejeitado.

### Timeout de poll (Kling)

O node aguarda ate `timeout_seconds` (padrao 600s) pelo video. Para videos longos
ou modelos lentos, aumentar o valor no workflow.


## Release: criar a tag

O runbook acima assume a tag publicada. Depois do merge na `main`:

```bash
git checkout main && git pull
git tag -a v0.2.0 -m 'AwGptImageNode (model livre) + AwKlingVideoNode'
git push origin v0.2.0
```

Sem essa etapa o documento acima aponta para uma referencia inexistente — foi
exatamente esse o estado em que este README nasceu, e vale conferir antes de mandar
alguem seguir o runbook.