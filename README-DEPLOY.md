# Deploy no ComfyUI (ECS/EFS)

Pack: `aw-comfyui-nodes` — node `GeminiImageInteractionsNode` (Gemini Interactions API).
Requer **v0.1.1 ou maior**. A v0.1.0 carrega sem erro e **nao registra node nenhum**
(ver "Erros conhecidos").

## Caminhos reais no host do ComfyUI

Confirme antes de qualquer comando — o pack e o Manager ficam dentro de `custom_nodes`:

```bash
ls -d /opt/content/custom_nodes
ls -d /opt/content/custom_nodes/ComfyUI-Manager
```

## Plano A — git clone direto (recomendado, nao usa o Manager)

O ComfyUI carrega no boot qualquer pasta de `custom_nodes` que tenha `__init__.py`.
O Manager e so uma conveniencia; o clone direto nao depende dele.

```bash
cd /opt/content/custom_nodes
git clone https://github.com/TrolljanO/aw-comfyui-nodes
cd aw-comfyui-nodes && git checkout v0.1.1

# dependencias (requests e python-dotenv normalmente ja existem no ComfyUI)
python -m pip install -r requirements.txt
```

Depois: rolling restart (secao abaixo).

## Plano B — ComfyUI-Manager

Pre-requisitos no `config.ini` do Manager. Localize o arquivo antes de editar:

```bash
find /opt/content -name config.ini -path '*Manager*'
```

```ini
[manager]
allow_git_url_install = true
security_level = normal-
```

`security_level = weak` tambem e aceito. Valores mais restritivos bloqueiam a
instalacao via URL.

- **Pela UI:** Manager → Install via Git URL → `https://github.com/TrolljanO/aw-comfyui-nodes` → Install
- **Pela CLI:**

```bash
python /opt/content/custom_nodes/ComfyUI-Manager/cm-cli.py \
    install https://github.com/TrolljanO/aw-comfyui-nodes
```

## Restart apos instalacao (ECS — DevOps executa)

O EFS e compartilhado entre replicas. Apos instalar, o DevOps deve fazer rolling
restart para carregar o novo pack em todos os containers.

**O pool do ComfyUI vive na conta PROD (`912668123297`, profile default do DevOps); o serviço `svc-confyui-prd-sqs` é o que atende `comfyui-hml` via `tg-comfyui-hml`. A conta homolog (`828435023027`) é só do Laravel:**

```bash
aws ecs update-service \
    --cluster cluster-confyui-prd \
    --service svc-confyui-prd-sqs \
    --force-new-deployment \
    --profile <aws-profile-prod>
```

Aguardar o deploy estabilizar (`aws ecs wait services-stable ...`) antes de verificar.

## Verificação

```bash
# 1. Confirmar que o pack aparece como instalado
# resposta é um dicionário { "<pasta do pack>": {ver, cnr_id, aux_id, enabled} }
curl -s -u "$COMFY_USER:$COMFY_PASS" "https://<COMFYUI_HOST>/v2/customnode/installed?mode=default" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); \
      print(d.get('aw-comfyui-nodes') or 'NAO ENCONTRADO')"

# 2. Confirmar que o node está registrado no schema  <-- esta é a prova que vale
curl -s http://<COMFYUI_HOST>/object_info/GeminiImageInteractionsNode \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys()))"
# Saída esperada: ['GeminiImageInteractionsNode']
# Saída {} = pack presente na pasta mas sem registrar node (ver "Erros conhecidos")

# 3. Validar o grafo do c11 sem custo (LoadImage com nome inexistente não executa)
curl -s -X POST http://<COMFYUI_HOST>/prompt \
    -H 'Content-Type: application/json' \
    -d '{"prompt":{"1":{"class_type":"LoadImage","inputs":{"image":"__nonexistent_validate__.png"}},"6":{"class_type":"GeminiImageInteractionsNode","inputs":{"prompt":"test","api_key":"GEMINI_API_KEY","model":"gemini-3-pro-image","aspect_ratio":"16:9","image_size":"2K","mime_type":"image/png","seed":0,"images":["1",0],"previous_interaction_id":"","system_instruction":""}}}}'
# Se o node não existe: response contém "error" com "node_errors"
# Se o node existe: response contém "prompt_id" (o LoadImage inexistente falhará na
#   execução, não na validação do schema)
```

## Variável de ambiente obrigatória

`GEMINI_API_KEY` deve existir no ambiente do processo ComfyUI.
No ECS, adicionar à task definition do serviço antes do rolling restart.

## Erros conhecidos

### `SyntaxError: from __future__ import annotations must be at the beginning of the file`

Traceback termina em `ComfyUI-Manager/glob/node_package.py`. **Nao e deste pack:**
o erro ocorre ao importar o proprio Manager, antes de a URL do repo ser lida.
Qualquer URL (ou nenhuma) produz o mesmo erro.

Confirmar e restaurar:

```bash
cd /opt/content/custom_nodes/ComfyUI-Manager
head -30 glob/node_package.py        # upstream tem o from __future__ na LINHA 1
git status --short                   # o arquivo aparece como modificado
git diff -- glob/node_package.py
git checkout -- glob/node_package.py # restaura a versao do upstream
python cm-cli.py show installed      # deve rodar sem SyntaxError
```

Enquanto o Manager estiver quebrado, use o **Plano A** — ele nao depende do Manager.

### `/object_info/GeminiImageInteractionsNode` devolve `{}` sem erro no log

Pack na versao **v0.1.0**. O `__init__.py` usava `from nodes import ...` (absoluto),
e a raiz do ComfyUI, que esta no `sys.path`, tem um `nodes.py` proprio — o registro
core. O import resolvia para o core, o pack carregava "com sucesso" e registrava
zero nodes. Corrigido na v0.1.1 (import relativo) e coberto por
`tests/test_comfyui_entrypoint.py`.

```bash
cd /opt/content/custom_nodes/aw-comfyui-nodes && git fetch --tags && git checkout v0.1.1
```
Seguido de rolling restart.
