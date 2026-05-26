# LangchainPreRAG

Este documento registra os procedimentos utilizados para construir o pipeline atual de pre-processamento, indexacao e consulta do RAG do projeto, com foco em orientacoes praticas para mulheres em situacao de violencia.

## Objetivo

Transformar documentos tecnicos e juridicos em uma base consultavel que privilegie:

- medidas protetivas
- direitos previstos na Lei Maria da Penha
- canais de ajuda
- telefones e contatos relevantes
- servicos de acolhimento e encaminhamento
- acoes praticas que a vitima pode executar

O objetivo nao foi indexar o texto bruto diretamente como base final, mas filtrar o conteudo para manter apenas orientacoes uteis para a usuaria.

## Pasta de origem

Os documentos de referencia utilizados no pre-processamento estao em:

- docs/Referencial_rag

Esses arquivos estao em formato TXT e incluem material tecnico, normativo, institucional e academico.

## Etapa 1: limpeza semantica com LLM

Script utilizado:

- src/rag/build_rag_base.py

Finalidade:

- ler os arquivos TXT da pasta de referencia
- dividir o conteudo em chunks
- enviar cada chunk para uma LLM via LangChain
- manter apenas trechos com orientacoes praticas diretamente aplicaveis a vitima
- descartar trechos irrelevantes retornados como `REMOVER`

### Carregamento

O pipeline tenta usar `TextLoader` do ecossistema LangChain. Quando essa dependencia nao esta disponivel, faz fallback para leitura nativa do arquivo com `utf-8` e, se necessario, `latin-1`.

### Chunking

Configuracao adotada no pipeline de limpeza:

- `chunk_size=2000`
- `chunk_overlap=200`

Quando `RecursiveCharacterTextSplitter` nao esta disponivel no ambiente, o script usa um splitter local por caractere com a mesma intencao funcional.

### Prompt de filtro e sintetizacao

O filtro semantico foi feito com prompt restritivo, pedindo para manter apenas:

- instrucoes praticas
- direitos
- canais de ajuda
- telefones
- orgaos de atendimento
- acoes diretas

E descartar:

- estatisticas
- debates academicos
- dotacoes orcamentarias
- linguagem institucional generica
- fluxos internos para servidores
- expedientes, creditos, avisos legais e textos sem orientacao concreta

Quando o chunk nao tinha valor pratico para a vitima, a LLM deveria responder exatamente:

- `REMOVER`

### Modelo utilizado na limpeza

Modelo configurado no script:

- `gpt-4o-mini`

Esse modelo foi acionado via `ChatOpenAI`, considerando não haver dados sensíveis, usando a variavel `OPENAI_API_KEY` carregada do arquivo `.env`.

### Saidas geradas

Arquivos produzidos pela limpeza semantica:

- data/processed/rag/base_rag_limpa.txt
- data/processed/rag/base_rag_limpa.jsonl

O arquivo `.txt` consolida as sinteses aprovadas.

O arquivo `.jsonl` preserva rastreabilidade por origem:

- arquivo de origem
- `chunk_id`
- sintese mantida

## Etapa 2: indexacao vetorial

Script utilizado:

- src/rag/index_rag_store.py

Finalidade:

- ler a base limpa consolidada
- dividir novamente em chunks menores para retrieval
- gerar embeddings vetoriais
- construir o indice FAISS

### Fonte usada na indexacao final

A indexacao final correta foi feita a partir de:

- data/processed/rag/base_rag_limpa.txt

Isso foi importante porque a indexacao inicial dos documentos brutos gerava muitos chunks irrelevantes, como:

- creditos institucionais
- avisos legais
- trechos de expediente
- material sem utilidade direta para acolhimento

### Chunking da indexacao

Configuracao usada:

- `chunk_size=1200`
- `chunk_overlap=150`

Esse segundo corte foi usado para melhorar a recuperacao vetorial e reduzir mistura excessiva de temas em um mesmo chunk.

### Backend de embeddings

Backend final utilizado:

- `openai`

Modelo configurado:

- `text-embedding-3-small`

O indice vetorial foi armazenado com FAISS.

### Saidas geradas pela indexacao

Arquivos gerados em:

- data/processed/rag/faiss_index

Artefatos principais:

- `index.faiss`
- `documents.jsonl`
- `manifest.json`

O `documents.jsonl` do indice final contem apenas chunks da base limpa, nao mais os documentos brutos originais.

## Etapa 3: consulta RAG

Script utilizado:

- src/rag/query_rag.py

Finalidade:

- receber uma pergunta da usuaria
- recuperar os chunks mais relevantes via FAISS
- montar contexto
- gerar resposta com LLM

### Retrieval

Configuracao principal:

- `top_k=5`
- backend de embedding: `openai`
- modelo de embedding: `text-embedding-3-small`

### Geracao da resposta

Backend principal usado:

- `openai`

Modelo configurado:

- `gpt-4o-mini`

Prompt de resposta orientado para:

- linguagem clara e empatica
- foco em seguranca imediata
- medidas protetivas
- canais oficiais de ajuda
- nao inventar telefones, enderecos ou procedimentos
- declarar explicitamente quando a base nao trouxer informacao suficiente

### Estrutura esperada da resposta

O formato solicitado ao modelo foi:

1. O que fazer agora
2. Direitos e medidas protetivas cabiveis
3. Canais de ajuda e onde procurar atendimento

## Credenciais e processamento

O pipeline RAG foi configurado para usar a OpenAI API em vez de processamento local de GPU.

Variavel usada:

- `OPENAI_API_KEY`

Arquivo:

- .env

Uso da chave:

- limpeza semantica dos chunks
- geracao de embeddings
- resposta final da consulta

Isso evita consumir GPU local nas etapas do RAG.

## Procedimento operacional executado

Sequencia efetivamente usada:

1. Corrigir o carregamento da `OPENAI_API_KEY` no `.env`.
2. Executar `src/rag/build_rag_base.py` para produzir a base limpa.
3. Verificar se os resultados em `base_rag_limpa.jsonl` estavam coerentes com o objetivo de acolhimento.
4. Reindexar com `src/rag/index_rag_store.py` usando `base_rag_limpa.txt` como entrada.
5. Validar com `src/rag/query_rag.py` usando pergunta realista e exibicao das fontes recuperadas.

## Problemas encontrados durante a construcao

Durante a implementacao, ocorreram os seguintes pontos:

1. O ambiente nao tinha alguns modulos auxiliares do ecossistema LangChain, exigindo fallbacks locais para carregamento e splitting.
2. A variavel da OpenAI no `.env` estava com nome incorreto inicialmente, impedindo autenticacao.
3. A primeira indexacao foi feita sobre material bruto, o que gerou baixa pertinencia nos chunks.
4. A qualidade melhorou somente depois da limpeza semantica e reindexacao da base filtrada.

## Estado atual do RAG

O estado considerado correto neste momento e:

- limpeza semantica executada
- base limpa gerada
- indice FAISS reconstruido a partir da base limpa
- consulta validada com OpenAI

Arquivos de referencia do estado atual:

- data/processed/rag/base_rag_limpa.txt
- data/processed/rag/base_rag_limpa.jsonl
- data/processed/rag/faiss_index/index.faiss
- data/processed/rag/faiss_index/documents.jsonl
- data/processed/rag/faiss_index/manifest.json

## Melhorias futuras recomendadas

Para evoluir o RAG depois desta etapa inicial:

1. Deduplicar orientacoes repetidas na base limpa.
2. Agrupar trechos por tema, por exemplo:
   - medidas protetivas
   - canais de ajuda
   - atendimento de saude
   - violencia sexual
   - defensoria e ministerio publico
3. Enriquecer metadados com jurisdicao, tipo de servico e nivel de urgencia.
4. Adicionar reranking para melhorar a selecao final dos chunks.
5. Integrar esse RAG com a resposta da LLM fine-tuned para complementar classificacao com orientacao juridica e protetiva.

## Comandos principais

Limpeza semantica:

```powershell
c:/Desenvolvimento/tech-challenge/tech-challenge-3/.venv/Scripts/python.exe src/rag/build_rag_base.py
```

Indexacao vetorial:

```powershell
c:/Desenvolvimento/tech-challenge/tech-challenge-3/.venv/Scripts/python.exe src/rag/index_rag_store.py --input-file data/processed/rag/base_rag_limpa.txt --embedding-backend openai --index-dir data/processed/rag/faiss_index
```

Consulta:

```powershell
c:/Desenvolvimento/tech-challenge/tech-challenge-3/.venv/Scripts/python.exe src/rag/query_rag.py --question "Estou sofrendo violencia domestica e ameacas. Quais medidas protetivas posso pedir e quais canais devo procurar agora?" --embedding-backend openai --answer-backend openai --show-sources
```