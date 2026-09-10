# Deploy no ComfyUI (ECS/EFS)

## Pré-requisitos no ComfyUI-Manager

No arquivo `config.ini` do Manager (normalmente em
`/opt/content/ComfyUI-Manager/config.ini` no EFS), as seguintes opções devem
estar presentes:

```ini
[manager]
allow_git_url_install = true
security_level = normal-
```

`security_level = weak` também é aceito. Valores mais restritivos bloqueiam
a instalação via URL.

## Instalação

### Via Manager UI (acesso ao painel ComfyUI)

1. Abrir **Manager → Install via Git URL**
2. Colar a URL: `https://github.com/TrolljanO/aw-comfyui-nodes`
3. Clicar **Install**

### Via cm-cli (linha de comando no container)

```bash
python /opt/content/ComfyUI-Manager/cm-cli.py \
    install https://github.com/TrolljanO/aw-comfyui-nodes
```

## Restart após instalação (ECS — DevOps executa)

O EFS é compartilhado entre réplicas. Após instalar, o DevOps deve fazer
rolling restart para carregar o novo pack em todos os containers:

**Conta prod (`828435023027`), profile `trajano-homolog` → ajustar para prod:**
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
curl -s http://<COMFYUI_HOST>/v2/customnode/installed?mode=default \
    | python3 -c "import sys,json; d=json.load(sys.stdin); \
      print([n for n in d.get('Nodes',[]) if 'aw-comfyui-nodes' in n.get('id','')])"

# 2. Confirmar que o node está registrado no schema
curl -s http://<COMFYUI_HOST>/object_info/GeminiImageInteractionsNode \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys()))"
# Saída esperada: ['GeminiImageInteractionsNode']

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
