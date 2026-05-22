# Arquivo criado para realizar o pré processamento de dados de ocorrência como eliminação de colunas e duplicatas

import pandas as pd

# carregar arquivo xlsx
df = pd.read_excel("data/Dados_Treinamento/DEAM2026.xlsx")

# Eliminar linhas campo HISTÓRICO = BLOQUEADO
df = df[df["Histórico"] != "BLOQUEADO"]

# Eliminar linhas campo Natureza = Em apuração
df = df[df["Natureza"] != "EM APURACAO"]

# Manter apenas um registro com mesmo:
# Ano de Registro, Unidade Policial de Registro, Número e Aditamento
df = df.drop_duplicates(
    subset=["Ano de Registro", "Unidade Policial de Registro", "Número", "Aditamento"]
)

# Entre registros com mesmo Ano de Registro, Unidade Policial de Registro e Número,
# manter o maior Aditamento
df = (
    df.sort_values("Aditamento", ascending=False)
      .drop_duplicates(subset=["Ano de Registro", "Unidade Policial de Registro", "Número"])
)

# eliminar colunas desnecessárias (depois da deduplicação)
colunas_para_eliminar = [
    "Unidade Policial de Registro", "Número", "Aditamento", "Crime Tentado (S/N)?", "Unidade de Apuração",
    "Sequencial", "Cd.Ocorrência", "Ano de Registro", "Cd.Unidade Registro",
    "Unidade Móvel", "Data do Registro", "Ano do Fato", "Data Início do Fato",
    "Flagrante (S/N)?", "Cd.Natureza", "Natureza Padronizada", "Cidade do Endereço do Fato",
    "Cidade com RA", "Área do Endereço do Fato", "Quadra do Endereço do Fato",
    "Complemento do Endereço do Fato", "Latitude GEO", "Longitude GEO",
    "Latitude UTM", "Longitude UTM", "Cd.Unidade Apuração", "Unidade Policial de Apuração", "Ano Proced.",
    "Data Instauração", "Cd. Órgão Proced.", "Órgão Procedimento", "Número Procedimento",
    "Cd.Tipo Proced.", "Tipo Procedimento", "TCNet?(Sim/Não)", "Tipo Instauração",
    "Data relatamento", "Nome Envolvido Proced", "Número do processo",
    "Órgão distribuição", "Retombamento SIM_NÂO?", "Data Indiciamento", "Código Envolvido Proced",
    "Incidência", "Incidência combinada"
]

df = df.drop(columns=colunas_para_eliminar, errors="ignore")

# Salvar o arquivo resultante
df.to_excel("data/Dados_Treinamento/DEAM2026_preprocessado.xlsx", index=False)