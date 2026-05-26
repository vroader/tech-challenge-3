# Tech Challenge 3 — Assistente Virtual de Apoio à Mulher em Situação de Violência

Assistente virtual especializado em segurança da mulher, com fine-tuning local de LLM, fluxos de decisão via LangGraph e orientações sobre serviços sociais e medidas jurídicas/protetivas, respeitando a privacidade e a sensibilidade do domínio.

---

## Arquitetura

```
Usuária digita mensagem
        │
        ▼
┌───────────────────┐
│  Classificador    │  Qwen2.5-7B + LoRA (GPU local)
│  fine-tuned       │  OU gpt-4o-mini via API (modo API-only)
│  30 categorias    │
└────────┬──────────┘
         │  categoria + confiança
         ▼
┌───────────────────────────────┐
│  LangGraph StateGraph         │
│                               │
│  SEM_VIOLÊNCIA ──► friend     │  gpt-4o-mini — conversa de apoio
│  Violência/Crime ──► rag      │  gpt-4o-mini + RAG — orientação jurídica
│                               │       (base: legislação + serviços de apoio)
│  ── log ──► MongoDB ──► END   │
└───────────────────────────────┘
        │
        ▼
  Interface Streamlit
  (sidebar com diagnóstico em tempo real)
```

---

## Dois modos de execução

### Modo completo (recomendado para avaliação do protótipo)

Exige GPU com pelo menos 16 GB de VRAM e o adapter LoRA treinado.
O classificador local roda no hardware do usuário, sem enviar o texto para nenhuma API.

```
USE_LOCAL_CLASSIFIER=true   # padrão
```

### Modo API-only (recomendado para rodar sem GPU)

Nenhum modelo local é carregado. A classificação é delegada ao `gpt-4o-mini` (zero-shot),
mantendo todo o fluxo funcional. Exige apenas uma chave OpenAI válida.

```
USE_LOCAL_CLASSIFIER=false
```

---

## Aviso importante — privacidade dos dados de treinamento

O adapter LoRA (`models/`) foi treinado em boletins de ocorrência reais que contêm
dados pessoais de vítimas de violência (nomes, endereços, contextos). Embora os pesos
do adapter não armazenem o texto em formato legível, pesquisas de privacidade em LLMs
demonstram que modelos fine-tunados podem memorizar fragmentos do treinamento e
reproduzi-los via prompts adversariais.

Por esses motivos:

- **Os arquivos `*.safetensors`, `*.bin` e checkpoints da pasta `models/` não são
  versionados neste repositório** (`.gitignore` os exclui).
- Os dados brutos de treinamento (`data/Dados_Treinamento/` e `data/processed/`) também
  são excluídos do versionamento.
- Para reproduzir o treinamento, é necessário obter os dados diretamente junto à equipe
  ou seguir o fluxo de preparação descrito abaixo.
- O projeto pode ser executado **sem o adapter** usando `USE_LOCAL_CLASSIFIER=false`,
  o que não apresenta nenhum risco de vazamento.

---

## Setup rápido

**Requisitos:** Python 3.11+, ambiente virtual ativo, Docker Desktop (para MongoDB).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copie o arquivo de exemplo e preencha as variáveis:

```bash
cp .env.example .env
# Edite .env com sua OPENAI_API_KEY
```

Suba o MongoDB:

```bash
docker-compose up -d mongo
```

Inicie a aplicação:

```bash
streamlit run app/streamlit_app.py
```

Para rodar sem GPU (modo API-only), adicione ao `.env`:

```
USE_LOCAL_CLASSIFIER=false
```

---

## Estrutura principal

| Caminho | Descrição |
| --- | --- |
| `app/streamlit_app.py` | Interface Streamlit (chat + sidebar de diagnóstico) |
| `src/chat/graph.py` | LangGraph StateGraph (classify → route → friend\|rag → log) |
| `src/chat/classifier_node.py` | Classificador local (LoRA) com fallback API-only |
| `src/chat/friend_node.py` | Resposta empática via gpt-4o-mini (modo WhatsApp) |
| `src/chat/rag_node.py` | Orientação jurídica via gpt-4o-mini + FAISS RAG |
| `src/chat/mongo_logger.py` | Log assíncrono de conversas no MongoDB |
| `src/data_prep/` | Scripts de preparação e geração do dataset de treino |
| `src/llm_local/` | Pipeline de treino (LoRA), inferência e avaliação local |
| `docs/Referencial_rag/` | Base documental para RAG (legislação, portarias, guias) |
| `data/processed/rag/` | Índice FAISS gerado a partir dos documentos de referência |
| `models/` | Saída do adapter LoRA treinado (não versionado — ver aviso acima) |

---

## Pipeline de fine-tuning local

### 1) Preparação do corpus

```bash
python src/data_prep/prepare_finetune_corpus.py \
    --input data/Dados_Treinamento/DEAM_2026_preprocessadoII.xlsx \
    --output-dir data/processed/finetune
```

### 2) Geração do dataset de treino

```bash
python src/llm_local/prepare_category_jsonl.py \
    --input data/processed/finetune/dataset_vitima_only_merged.csv \
    --output-dir data/processed/finetune/llm_local
```

Saídas: `train.jsonl`, `val.jsonl`, `test.jsonl`, `categories.json`

### 3) Fine-tuning (LoRA)

Modelo base: `Qwen/Qwen2.5-7B-Instruct` | Adapter: LoRA r=16, alpha=32
Dataset: ~2346 exemplos de treino / 30 categorias / 2 épocas

```bash
python src/llm_local/train_local_lora.py
```

Opções: `--cache-dir .hf_cache` | `--allow-remote` | `--use-4bit`

Saída: `models/qwen2.5-7b-vitima-only-lora/`

Tempo estimado na RTX 5080 (16 GB): **~50–80 minutos**

### 4) Inferência com o adapter treinado

```bash
python src/llm_local/predict_local.py \
    --text "Narrativa para classificar"
```

### 5) Avaliação

```bash
python src/llm_local/evaluate_local.py --limit 100
```

---

## Construção do índice RAG

```bash
python src/rag/build_index.py \
    --docs-dir docs/Referencial_rag \
    --output-dir data/processed/rag
```

O índice FAISS gerado (documentos públicos) está versionado no repositório.

---

## Variáveis de ambiente (`.env`)

| Variável | Obrigatória | Descrição |
| --- | --- | --- |
| `OPENAI_API_KEY` | Sim | Chave da API OpenAI |
| `MONGO_URI` | Não | URI do MongoDB (padrão: `mongodb://localhost:27017`) |
| `USE_LOCAL_CLASSIFIER` | Não | `true` = usa modelo local (padrão) / `false` = modo API-only |

---

## Aplicação Streamlit — funcionalidades

- **Chat de apoio** com histórico de conversa
- **Sidebar de diagnóstico** em tempo real: categoria detectada, confiança do classificador e caminho tomado no grafo (exibido apenas para avaliação do protótipo)
- **"Nova conversa"** reinicia sessão e limpa diagnóstico
- **Expander "Sobre os modelos"** explica a arquitetura ao avaliador

---

## Aviso operacional

Se o fine-tuning local estiver em execução, não interromper o processo.
Não fechar o terminal de treino, não reiniciar a máquina e não matar processos Python/torch durante a execução.

---

## Integrantes

| Nome |
| --- |
| Marcelo Arruda de Siqueira |
| Leonardo Barbosa Nogueira |
| Jose Flavio Neto |
| Pedro Matias dos Santos |
| Wellington Vieira de Oliveira |
