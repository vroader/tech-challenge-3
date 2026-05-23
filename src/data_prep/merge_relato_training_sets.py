import argparse
from pathlib import Path

import pandas as pd


def normalize_relatos(df: pd.DataFrame, start_row_id: int) -> pd.DataFrame:
    # Only the first two columns are merged (Classe, Relato).
    relatos_2cols = df.iloc[:, :2].copy()
    relatos_2cols.columns = ["Classe", "Relato"]

    normalized = pd.DataFrame(
        {
            "row_id": range(start_row_id, start_row_id + len(relatos_2cols)),
            "natureza": relatos_2cols["Classe"].fillna("").astype(str).str.strip(),
            "text_anonymized": relatos_2cols["Relato"].fillna("").astype(str).str.strip(),
            "segment_count": 1,
        }
    )
    normalized = normalized[normalized["text_anonymized"] != ""].reset_index(drop=True)
    return normalized


def merge_dataset(base_path: Path, relatos_df: pd.DataFrame, output_path: Path) -> int:
    base_df = pd.read_csv(base_path, encoding="utf-8-sig")
    required_base_cols = {"row_id", "natureza", "text_anonymized", "segment_count"}
    missing_base_cols = required_base_cols - set(base_df.columns)
    if missing_base_cols:
        raise ValueError(f"Colunas ausentes em dataset_vitima_only.csv: {sorted(missing_base_cols)}")

    max_row_id = int(pd.to_numeric(base_df["row_id"], errors="coerce").fillna(-1).max())
    normalized_relatos = normalize_relatos(relatos_df, max_row_id + 1)

    merged_df = pd.concat([base_df, normalized_relatos], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(merged_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mescla relatos_treinamento.csv ao dataset vitima_only."
    )
    parser.add_argument(
        "--relatos-input",
        default="data/processed/finetune/relatos_treinamento.csv",
        help="CSV de relatos (serao usadas apenas as duas primeiras colunas).",
    )
    parser.add_argument(
        "--vitima-only-input",
        default="data/processed/finetune/dataset_vitima_only.csv",
        help="Dataset base vitima_only.",
    )
    parser.add_argument(
        "--vitima-only-output",
        default="data/processed/finetune/dataset_vitima_only_merged.csv",
        help="Saida do dataset vitima_only mesclado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    relatos_path = Path(args.relatos_input)
    vitima_only_input = Path(args.vitima_only_input)
    vitima_only_output = Path(args.vitima_only_output)

    for path in [relatos_path, vitima_only_input]:
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    relatos_df = pd.read_csv(relatos_path, encoding="utf-8-sig")
    if relatos_df.shape[1] < 2:
        raise ValueError("relatos_treinamento.csv precisa ter pelo menos duas colunas (Classe e Relato).")

    vitima_only_count = merge_dataset(vitima_only_input, relatos_df, vitima_only_output)

    print(f"[OK] vitima_only merged: {vitima_only_output} ({vitima_only_count} linhas)")


if __name__ == "__main__":
    main()