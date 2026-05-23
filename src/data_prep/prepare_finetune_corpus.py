import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ADMIN_KEYWORDS = {
    "encaminhamento",
    "oficio",
    "ofício",
    "procedimento",
    "procedimentos",
    "providencia",
    "providência",
    "administrativo",
    "administrativa",
    "cartorio",
    "cartório",
    "protocolo",
    "inquerito",
    "inquérito",
    "boletim",
    "registro",
    "distribuicao",
    "distribuição",
    "juntada",
    "autuacao",
    "autuação",
    "despacho",
    "expediente",
    "notificacao",
    "notificação",
    "diligencia",
    "diligência",
    "audiencia",
    "audiência",
    "delegacia",
    "plantao",
    "plantão",
    "memorando",
    "guarnicao",
    "guarnição",
    "iml",
    "autoridade policial",
    "dp de apuracao",
    "dp de apuração",
}

VICTIM_HINTS = {
    "vitima",
    "vítima",
    "declarante",
    "ofendida",
    "comunicante",
}

OTHER_PARTIES_HINTS = {
    "autor",
    "agressor",
    "testemunha",
    "vizinho",
    "vizinha",
    "pai",
    "mae",
    "mãe",
    "filho",
    "filha",
    "irmão",
    "irmao",
    "irmã",
    "irma",
}

ROLE_PATTERNS = {
    "VITIMA": re.compile(
        r"(?i)(v[ií]tima|ofendida|declarante|comunicante)\s*[:\-]\s*([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+){0,4})"
    ),
    "AUTOR": re.compile(
        r"(?i)(autor(?:a)?|agressor(?:a)?)\s*[:\-]\s*([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+){0,4})"
    ),
    "TESTEMUNHA": re.compile(
        r"(?i)(testemunha)\s*[:\-]\s*([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+){0,4})"
    ),
    "POLICIAL": re.compile(
        r"(?i)(policial|agente|delegad[oa])\s*[:\-]\s*([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+){0,4})"
    ),
}

GENERIC_PATTERNS = [
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF]"),
    (re.compile(r"\b\d{11}\b"), "[CPF]"),
    (re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"), "[CNPJ]"),
    (re.compile(r"\b\d{14}\b"), "[CNPJ]"),
    (re.compile(r"[\w.\-+]+@[\w\-]+\.[\w\-.]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}-?\d{4}\b"), "[TELEFONE]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATA]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATA]"),
    (re.compile(r"\b(?:rua|av\.?|avenida|quadra|q\.|conjunto|lote|bloco|apto|apartamento)\b[^,.;\n]*", re.IGNORECASE), "[ENDERECO]"),
    (re.compile(r"\b(?:processo|procedimento|boletim|inquerito|inquérito|protocolo)\s*(?:n[oº.]?\s*)?[\w./-]+", re.IGNORECASE), "[ID_ADMIN]"),
        (re.compile(r"\b(?:compareceu(?:\s+a)?\s+(?:esta|nesta)?\s*delegacia\s*)([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç'`-]+){1,5})", re.IGNORECASE), "compareceu a delegacia [NOME]"),
        (re.compile(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,}(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,}){1,5}\b"), "[NOME]"),
]

SEGMENT_SPLIT = re.compile(r"[\n\r]+|(?<=[.;!?])\s+")
MULTISPACE = re.compile(r"\s+")
PLACEHOLDER_STRIP_RE = re.compile(r"\[(?:NOME|ENDERECO|DATA)\]")
EMPTY_PUNCT_RE = re.compile(r"^[\W_]+$")

OITIVA_START_RE = re.compile(
    r"(?is)\b(?:oitiva(?:\(s\))?|oitivas?)\b\s*[:\-]?"
)

# The end marker closes the current Oitiva block and excludes administrative content.
OITIVA_END_RE = re.compile(
    r"(?is)\b(?:"
    r"das?\s+provid[eê]ncias?|provid[eê]ncias?|aditamento(?:\s*n?[ºo.]?\s*\d+)?|"
    r"despacho|consigno|informo|dos\s+fatos|encaminhamento|"
    r"certid[aã]o|laudo|representa[cç][aã]o|requerimento"
    r")\b\s*:?"
)


@dataclass
class PreparedRow:
    row_id: int
    natureza: str
    text_anonymized: str
    segment_count: int


def normalize_text(value: str) -> str:
    text = str(value or "").strip()
    text = MULTISPACE.sub(" ", text)
    return text


def fold_text(value: str) -> str:
    text = normalize_text(value)
    folded = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    return folded.lower()


def split_segments(text: str) -> List[str]:
    parts = [normalize_text(chunk) for chunk in SEGMENT_SPLIT.split(text)]
    return [chunk for chunk in parts if chunk]


def classify_segment(segment: str) -> str:
    lowered = segment.lower()

    if any(keyword in lowered for keyword in VICTIM_HINTS):
        return "victim"

    if any(keyword in lowered for keyword in OTHER_PARTIES_HINTS):
        return "other"

    if " declarou" in lowered or " relatou" in lowered or " informou" in lowered:
        return "victim"

    if any(keyword in lowered for keyword in ADMIN_KEYWORDS):
        return "admin"

    return "unknown"


def extract_oitiva_block(history: str) -> str:
    """
    Keep only content that is explicitly under Oitiva(s) sections.
    This enforces exclusion of administrative sections such as Providencias.
    """
    raw = str(history or "")
    if not raw.strip():
        return ""

    starts = list(OITIVA_START_RE.finditer(raw))
    if not starts:
        return ""

    chunks: List[str] = []
    for i, start in enumerate(starts):
        tail = raw[start.end():]
        end = OITIVA_END_RE.search(tail)
        if end:
            block = tail[:end.start()]
        else:
            next_start = starts[i + 1].start() if i + 1 < len(starts) else len(raw)
            block = raw[start.end():next_start]

        block = block.strip(" :-\n\r\t")
        if block:
            chunks.append(block)

    return "\n".join(chunks)


def apply_generic_redactions(text: str) -> str:
    redacted = text
    for pattern, replacement in GENERIC_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def role_pseudonymize(text: str, role_counters: Dict[str, int], role_map: Dict[Tuple[str, str], str]) -> str:
    redacted = text

    for role, pattern in ROLE_PATTERNS.items():
        def replace_match(match: re.Match) -> str:
            raw_name = normalize_text(match.group(2))
            key = (role, raw_name.lower())
            if key not in role_map:
                role_counters[role] = role_counters.get(role, 0) + 1
                role_map[key] = f"[{role}_{role_counters[role]}]"
            return f"{match.group(1)}: {role_map[key]}"

        redacted = pattern.sub(replace_match, redacted)

    return redacted


def anonymize_text(text: str) -> str:
    role_counters: Dict[str, int] = {}
    role_map: Dict[Tuple[str, str], str] = {}
    redacted = role_pseudonymize(text, role_counters, role_map)
    redacted = apply_generic_redactions(redacted)
    return normalize_text(redacted)


def cleanup_output_text(text: str) -> str:
    cleaned = PLACEHOLDER_STRIP_RE.sub(" ", str(text or ""))
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?]){2,}", r"\1", cleaned)
    cleaned = MULTISPACE.sub(" ", cleaned).strip(" ,.;:!?-\n\r\t")
    if not cleaned or EMPTY_PUNCT_RE.match(cleaned):
        return ""
    return cleaned


def choose_column(df: pd.DataFrame, options: List[str]) -> str:
    available = {col.lower(): col for col in df.columns}
    for opt in options:
        if opt.lower() in available:
            return available[opt.lower()]
    raise ValueError(f"Nenhuma coluna encontrada entre: {options}")


def load_dataframe(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    raise ValueError(f"Formato de entrada nao suportado: {input_path.suffix}")


def build_datasets(df: pd.DataFrame, history_col: str, nature_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    victim_rows: List[PreparedRow] = []

    stats = {
        "rows_input": int(len(df)),
        "rows_with_history": 0,
        "rows_with_oitiva": 0,
        "rows_without_oitiva": 0,
        "segments_total": 0,
        "segments_admin": 0,
        "segments_victim": 0,
        "segments_other": 0,
        "segments_unknown": 0,
        "rows_victim_output": 0,
    }

    for idx, row in df.iterrows():
        history_raw = normalize_text(row.get(history_col, ""))
        nature = normalize_text(row.get(nature_col, ""))

        # Remove registros em que o campo historico esteja vazio ou so com espacos.
        if not history_raw:
            continue

        stats["rows_with_history"] += 1

        oitiva_text = extract_oitiva_block(history_raw)
        if not oitiva_text:
            stats["rows_without_oitiva"] += 1
            continue

        stats["rows_with_oitiva"] += 1

        segments = split_segments(oitiva_text)
        stats["segments_total"] += len(segments)

        victim_segments: List[str] = []
        for seg in segments:
            label = classify_segment(seg)
            stats[f"segments_{label}"] += 1

            if label == "admin":
                continue

            anonymized = cleanup_output_text(anonymize_text(seg))
            if not anonymized:
                continue

            if label == "victim":
                victim_segments.append(anonymized)

        if victim_segments:
            victim_text = cleanup_output_text(" ".join(victim_segments))
            if victim_text:
                victim_rows.append(
                    PreparedRow(
                        row_id=int(idx),
                        natureza=nature,
                        text_anonymized=victim_text,
                        segment_count=len(victim_segments),
                    )
                )

    stats["rows_victim_output"] = len(victim_rows)

    victim_df = pd.DataFrame([row.__dict__ for row in victim_rows])
    return victim_df, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara corpus anonimizado para fine-tuning local (vítima-only e vítima+contexto)."
    )
    parser.add_argument(
        "--input",
        default="data/Dados_Treinamento/DEAM_2026_preprocessado.xlsx",
        help="Arquivo de entrada (.xlsx, .csv ou .parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/finetune",
        help="Diretorio de saida",
    )
    parser.add_argument(
        "--history-col",
        default="Histórico",
        help="Nome da coluna de texto historico",
    )
    parser.add_argument(
        "--nature-col",
        default="Natureza",
        help="Nome da coluna da natureza da ocorrencia",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        fallback_path = Path("data/Dados_Treinamento/DEAM2026_preprocessado.xlsx")
        if input_path.name == "DEAM_2026_preprocessado.xlsx" and fallback_path.exists():
            input_path = fallback_path
            print(f"[AVISO] Entrada padrao nao encontrada. Usando arquivo existente: {input_path}")
        else:
            raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(input_path)
    history_col = choose_column(df, [args.history_col, "historico", "HISTORICO"])
    nature_col = choose_column(df, [args.nature_col, "Natureza", "Natureza Padronizada"])

    victim_df, stats = build_datasets(df, history_col, nature_col)

    victim_csv_path = output_dir / "dataset_vitima_only.csv"
    report_path = output_dir / "preparation_report.json"

    victim_df.to_csv(victim_csv_path, index=False)

    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)

    print(f"[OK] dataset vitima-only (csv): {victim_csv_path}")
    print(f"[OK] relatorio: {report_path}")


if __name__ == "__main__":
    main()
