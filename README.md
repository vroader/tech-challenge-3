# Tech Challenge 3 — Otimização do modelo preditor de recorrência em casos de violência contra a mulher

Desenvolver um assistente virtual de atendimento especializado em segurança da mulher, utilizando fine-tuning de LLMs com dados
específicos da área e implementando fluxos automatizados de decisão através do LangChain, orientações sobre serviços sociais e medidas 
jurídicas e protetivas  sempre respeitando protocolos de segurança, privacidade e sensibilidade cultural específicos do atendimento feminino.

## Setup rápido

Requisitos: **Python 3.11+** (recomendado).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Variáveis de ambiente (opcional):  no arquivo `.env` na raiz do projeto (o cliente usa `python-dotenv`).

**Dados:** Os dados para treinamento do fine tunning são sensíveis e não estão disponíveis no repositório seu tratamento 
está descrito em PreprocessamentoFineTune.md

## Estrutura do repositório

| Caminho                         | Descrição                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `app/streamlit_app.py`          | Interface Streamlit para carregar o frontend                                               |
| `data/Dados_Treinamento`        | Dados brutos para treinamento do finetune (gitignore)                                      |
| `data/processed`                | Dados processados para finetunning (gitignore)                                             |
| `docs/Referencial_rag`          | Documentos públicos coletados para configuração do rag                                     |
| `models/`                       | Modelos treinados                                                                          |
| `source/`                       | Códigos                                                                                    |

## Uso

### Aplicação Streamlit

```bash
streamlit run app/streamlit_app.py
```

O app permite escolher entre variantes de modelo em `models/` (conforme arquivos `.pkl` gerados pela execução do `src/optimize_ga.py`).

### Preparacao local para fine-tuning com dados sigilosos

Foi adicionada uma etapa inicial de preparo de corpus para uso 100% local:

1. Anonimizacao e separacao de narrativas relevantes (remove segmentos administrativos):

```bash
python src/data_prep/prepare_finetune_corpus.py \
	--input data/Dados_Treinamento/DEAM_2026_preprocessadoII.xlsx \
	--output-dir data/processed/finetune
```

O script extrai exclusivamente o conteudo do bloco `Oitiva(s)` e elimina secoes administrativas como `Das Providencias`, `Despacho` e `Aditamento`.
Caso o arquivo com underscore nao exista, o script usa automaticamente `data/Dados_Treinamento/DEAM2026_preprocessadoII.xlsx`.

Saidas principais em `data/processed/finetune`:
- `dataset_vitima_only.csv` (apenas segmentos classificados como fala da vitima)
- `dataset_vitima_contexto.csv` (vitima + contexto nao administrativo)
- `preparation_report.json` (metricas de extração)

2. Geracao de dataset de instrucoes com rotulo inicial de risco (heuristico):

- 

```bash
python src/data_prep/build_instruction_dataset.py \
	--input data/processed/finetune/dataset_vitima_contexto.csv \
	--output-dir data/processed/finetune
```

Saidas principais:
- `risk_train.jsonl`
- `risk_val.jsonl`
- `risk_label_stats.json`

Observacao: os rotulos de risco gerados sao baseline por regras e devem ser validados por especialistas antes de qualquer uso operacional.

### Modelo textual local para narrativas



```bash
python src/text_classifier.py train
python src/text_classifier.py predict --text "texto da narrativa aqui"
```

Saidas do treino:
- `models/text_risk_classifier.pkl`
- `models/text_risk_classifier_metrics.json`

Se quiser testar de forma interativa, rode apenas:

```bash
python src/text_classifier.py predict
```

O modelo de texto usa os arquivos `risk_train.jsonl` e `risk_val.jsonl` gerados na etapa de preparacao.

## Integrantes

| Nome                           |
| ------------------------------ |
| Marcelo Arruda de Siqueira     |
| Leonardo Barbosa Nogueira      |
| Jose Flavio Neto               |
| Pedro Matias dos Santos        |
| Wellington Vieira de Oliveira  |
