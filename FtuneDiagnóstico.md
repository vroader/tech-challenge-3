# Diagnostico do Fine-Tuning

## Objetivo

Este documento consolida o processo de fine-tuning local do classificador de narrativas de violencia contra a mulher, os problemas encontrados durante a execucao, as correcoes aplicadas e o estado atual do projeto. O foco aqui e exclusivamente o fluxo de fine-tune, inferencia e avaliacao do modelo local com LoRA.

## Escopo do que foi feito

O trabalho realizado envolveu quatro frentes principais:

1. Preparar o pipeline para rodar de forma offline ou local-first, sem depender de trafego externo durante a execucao normal.
2. Ajustar o treinamento para compatibilidade com a versao instalada de `transformers` e com o ambiente Windows.
3. Validar a inferencia e a avaliacao do adapter treinado.
4. Identificar a causa do resultado incorreto observado nos testes iniciais do modelo fine-tunado.

## Artefatos gerados

O fine-tuning produziu o adapter em `models/qwen2.5-7b-vitima-only-lora` com os seguintes artefatos relevantes:

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `run_config.json`
- `checkpoint-200/`
- `checkpoint-294/`
- `offload/`

Esses artefatos indicam que houve execucao efetiva do treinamento e persistencia dos checkpoints intermediarios e finais.

## Configuracao do treino executado

Com base em `models/qwen2.5-7b-vitima-only-lora/run_config.json`, a execucao consolidada do treino usou:

- Modelo base: `Qwen/Qwen2.5-7B-Instruct`
- Dataset de treino: `data/processed/finetune/llm_local/train.jsonl`
- Dataset de validacao: `data/processed/finetune/llm_local/val.jsonl`
- Saida: `models/qwen2.5-7b-vitima-only-lora`
- Tamanho maximo de sequencia: `1024`
- Batch size por dispositivo: `1`
- Gradient accumulation: `16`
- Learning rate: `2e-4`
- Epocas: `2.0`
- Seed: `42`
- `allow_remote=false`
- `use_4bit=false`

## Estrutura do pipeline

O fluxo operacional ficou assim:

1. Preparacao do corpus supervisionado em `data/processed/finetune`.
2. Conversao para JSONL de treino em `data/processed/finetune/llm_local`.
3. Treinamento LoRA local via `src/llm_local/train_local_lora.py`.
4. Inferencia pontual via `src/llm_local/predict_local.py`.
5. Avaliacao por conjunto de teste via `src/llm_local/evaluate_local.py`.

O resolvedor `src/llm_local/local_model.py` passou a ser o ponto central para localizar o modelo base a partir de pasta local ou cache do Hugging Face, evitando download automatico quando `allow_remote` esta desabilitado.

## Problemas encontrados durante o processo

### 1. Falhas de conectividade com Hugging Face

O ambiente apresentou falhas de DNS e conectividade externa durante a tentativa de carregar o modelo base e o tokenizer. Isso impactava treino, inferencia e avaliacao.

### 2. Necessidade de rodar sem trafego externo

Havia o requisito explicito de garantir execucao sem dependencia de rede durante o fine-tuning. Isso exigiu mudar o comportamento padrao do pipeline para offline/local-first.

### 3. Incompatibilidade com a API instalada do `transformers`

Foram observadas mudancas de API compativeis com a serie `transformers 5.9.0`, em especial:

- `TrainingArguments` usando `eval_strategy` em vez de `evaluation_strategy`
- `Trainer` usando `processing_class` em vez de `tokenizer`
- remocao de alguns argumentos antes aceitos em versoes anteriores

Sem esse ajuste, o script de treino nao ficava estavel no ambiente atual.

### 4. Problemas de carregamento do adapter com offload

Durante a avaliacao e a inferencia, tentativas de carregar o modelo usando estrategias automaticas de distribuicao e offload dispararam erros do `peft` e do `accelerate`, incluindo o erro abaixo:

```text
KeyError: 'base_model.model.model.lm_head'
```

Esse comportamento apareceu ao tentar misturar adapter LoRA com offload automatico em uma configuracao fragil para esse ambiente.

### 5. Resultado de avaliacao artificialmente zerado

Em uma avaliacao inicial com subconjunto de teste, a saida indicou:

- `accuracy=0.0000`
- `macro_f1=0.0000`

Isso sugeria, a principio, que o modelo havia treinado mal ou que o adapter nao estava sendo aplicado corretamente. Depois foi identificado que a causa principal era outra.

## Correcoes aplicadas

### 1. Pipeline offline/local-first

Os scripts passaram a respeitar execucao local por padrao, com:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- uso de `local_files_only=True` quando `allow_remote` nao e informado
- resolucao local do modelo base por `resolve_model_source(...)`

O objetivo foi impedir downloads implicitos e garantir previsibilidade operacional.

### 2. Ajustes do treinamento para a versao do ecossistema instalada

O script `src/llm_local/train_local_lora.py` foi adaptado para a API efetivamente disponivel no ambiente. Isso incluiu a configuracao correta do `TrainingArguments` e do `Trainer` para a versao de `transformers` em uso.

### 3. Estabilizacao do carregamento de inferencia e avaliacao

Os scripts `src/llm_local/predict_local.py` e `src/llm_local/evaluate_local.py` foram ajustados para carregar o modelo base e o adapter de forma mais simples, evitando a combinacao de offload automatico que estava provocando erros no `peft`.

Na pratica, o fluxo funcional passou a ser:

1. Carregar o modelo base.
2. Carregar o adapter LoRA com `PeftModel.from_pretrained(...)`.
3. Mover o conjunto para CUDA apenas quando disponivel.

### 4. Correcao do bug de pos-processamento na avaliacao e predicao

Esta foi a correcao mais importante da etapa atual.

O codigo estava decodificando a sequencia completa retornada por `generate(...)`, incluindo o prompt original. Como o prompt continha toda a lista de categorias permitidas, o fallback do parser encontrava uma categoria valida dentro do proprio prompt e acabava retornando, de forma enviesada, a primeira categoria reconhecida.

Na pratica, isso fazia o sistema tender a prever sempre:

```json
{"categoria": "ABANDONO DE INCAPAZ", "confianca": 0.0}
```

mesmo quando o texto nao tinha relacao com essa classe.

A correcao aplicada foi decodificar apenas os tokens novos gerados pelo modelo, descartando o trecho correspondente ao prompt de entrada.

Em termos logicos:

```python
prompt_len = inputs["input_ids"].shape[-1]
generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
```

Essa mudanca foi aplicada tanto em `src/llm_local/predict_local.py` quanto em `src/llm_local/evaluate_local.py`.

## Evidencias coletadas durante a validacao

### Predicao manual antes da correcao do decode

Uma narrativa com agressao fisica e ameaca retornou incorretamente:

```json
{"categoria": "ABANDONO DE INCAPAZ", "confianca": 0.0}
```

Isso era incoerente com o conteudo do relato e reforcou a suspeita de erro no parser ou no recorte da geracao.

### Diagnostico com amostras do conjunto de teste

Ao comparar amostras do `test.jsonl`, foi observado que diferentes exemplos estavam sendo mapeados para a mesma categoria indevida, inclusive exemplos rotulados como `SEM_VIOLÊNCIA` e `AMEACA`. Esse padrao reforcou que o problema nao era apenas de qualidade estatistica do modelo, mas de leitura incorreta da resposta gerada.

### Predicao manual apos a correcao

Depois da correcao do decode, a mesma entrada manual passou a retornar uma classe plausivel:

```json
{"categoria": "LESÃO CORPORAL", "confianca": 0.95}
```

Essa resposta deixou de colapsar na primeira categoria da lista e passou a refletir o conteudo da narrativa.

### Avaliacao rapida apos a correcao

Executando a avaliacao com `--limit 10`, o resultado mudou para:

- `accuracy=0.7000`
- `macro_f1=0.3143`

Resumo por classe observado nesse subconjunto:

- `SEM_VIOLÊNCIA`: desempenho forte no recorte avaliado
- `INJURIA`: houve acertos no subconjunto
- `AMEACA`, `DANO` e `VIAS DE FATO`: ainda com erros no recorte testado

Isso confirma que a metrica zerada anterior estava contaminada por um bug de pos-processamento.

## Estado atual do fine-tuning

No estado atual, o pipeline apresenta o seguinte quadro:

- O treinamento local gerou adapter, checkpoints e configuracao persistida.
- O fluxo esta preparado para operar em modo offline/local-first.
- A inferencia local funciona sem cair automaticamente na primeira categoria da lista.
- A avaliacao voltou a produzir metricas coerentes.
- Ainda existem erros reais de classificacao em parte das classes, que agora podem ser investigados de forma legitima.

Em outras palavras: o problema mais grave desta fase nao era apenas o fine-tune em si, mas um bug na etapa de leitura da resposta do modelo. Esse bug ja foi isolado e corrigido.

## O que ainda pode estar dando errado

Agora que a avaliacao nao esta mais contaminada pelo prompt completo, a proxima etapa de investigacao deve olhar para problemas reais de modelagem e dados. As hipoteses mais provaveis sao:

1. Desbalanceamento entre classes no conjunto de treinamento.
2. Sobreposicao semantica entre categorias proximas, como `AMEACA`, `INJURIA`, `VIAS DE FATO` e `LESÃO CORPORAL - VIOLÊNCIA DOMÉSTICA`.
3. Prompt de classificacao ainda muito rigido ou pouco informativo.
4. Ruido de rotulagem no dataset supervisionado.
5. Quantidade insuficiente de exemplos para certas classes.
6. Necessidade de validacao mais ampla do adapter em todo o conjunto de teste.

## Recomendacoes para a proxima investigacao

Quando formos investigar o que ainda esta errado, a ordem recomendada e:

1. Rodar a avaliacao completa em `test.jsonl`.
2. Gerar uma amostra `real x previsto` por classe.
3. Levantar matriz de confusao entre categorias proximas.
4. Verificar distribuicao de frequencia das classes em treino, validacao e teste.
5. Revisar exemplos rotulados das classes com pior desempenho.
6. Reavaliar prompt e formato de supervisao usados no JSONL.

## Conclusao executiva

O processo de fine-tuning local foi efetivamente executado e produziu um adapter valido. O maior problema diagnostico desta etapa foi um erro de pos-processamento na inferencia e na avaliacao: o sistema estava parseando o prompt junto com a resposta gerada, o que enviesava a classificacao para a primeira categoria encontrada na lista. Depois da correcao, a avaliacao deixou de ficar zerada e o modelo passou a apresentar comportamento coerente o suficiente para uma investigacao real de desempenho por classe.

Ou seja: antes de investigar qualidade do treinamento, primeiro foi necessario corrigir a forma como a resposta do modelo estava sendo lida. Essa etapa ja esta documentada e resolvida.