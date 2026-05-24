import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split


SYSTEM_PROMPT = (
    "Voce e uma classificadora especializada em narrativas de violencia contra a mulher. "
    "Classifique o texto em exatamente uma categoria da lista fornecida. "
    "Nao invente categoria. "
    "Responda apenas com JSON valido no formato: {\"categoria\":\"NOME_EXATO_DA_CATEGORIA\"}."
)


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def build_instruction_text(text: str, categories: List[str]) -> str:
    cat_block = "\n".join(categories)
    return (
        f"Categorias permitidas:\n{cat_block}\n\n"
        f"Texto para classificar:\n{text}\n"
    )


def to_sft_record(text: str, label: str, categories: List[str]) -> Dict[str, str]:
    user_prompt = build_instruction_text(text=text, categories=categories)
    assistant = json.dumps({"categoria": label}, ensure_ascii=False)
    full_text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_prompt}\n"
        f"<|assistant|>\n{assistant}"
    )
    return {
        "text": full_text,
        "input_text": text,
        "label": label,
    }


def write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte dataset_vitima_only_merged.csv para JSONL de treino local de LLM."
    )
    parser.add_argument(
        "--input",
        default="data/processed/finetune/dataset_vitima_only_merged.csv",
        help="CSV de entrada com colunas natureza e text_anonymized.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/finetune/llm_local",
        help="Diretorio de saida para train/val/test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fracao de teste.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fracao de validacao.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Limite de caracteres por texto para caber no contexto.",
    )
    return parser.parse_args()


def split_with_fallback(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    stratify_col: str,
):
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[stratify_col],
        )
    except ValueError as exc:
        print(f"[AVISO] Split estratificado indisponivel ({exc}). Usando split aleatorio.")
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required_cols = {"natureza", "text_anonymized"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

    work = df[["natureza", "text_anonymized"]].copy()
    work["natureza"] = work["natureza"].fillna("").astype(str).map(normalize_text)
    work["text_anonymized"] = work["text_anonymized"].fillna("").astype(str).map(normalize_text)
    work = work[(work["natureza"] != "") & (work["text_anonymized"] != "")].reset_index(drop=True)

    work["text_anonymized"] = work["text_anonymized"].map(lambda t: t[: args.max_chars])

    categories = sorted(work["natureza"].unique().tolist())

    train_val, test = split_with_fallback(
        df=work,
        test_size=args.test_size,
        seed=args.seed,
        stratify_col="natureza",
    )

    val_ratio_over_train_val = args.val_size / (1.0 - args.test_size)
    train, val = split_with_fallback(
        df=train_val,
        test_size=val_ratio_over_train_val,
        seed=args.seed,
        stratify_col="natureza",
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = [
        to_sft_record(text=row.text_anonymized, label=row.natureza, categories=categories)
        for row in train.itertuples(index=False)
    ]
    val_records = [
        to_sft_record(text=row.text_anonymized, label=row.natureza, categories=categories)
        for row in val.itertuples(index=False)
    ]
    test_records = [
        {"input_text": row.text_anonymized, "label": row.natureza}
        for row in test.itertuples(index=False)
    ]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"
    meta_path = out_dir / "categories.json"

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)

    metadata = {
        "input": str(input_path),
        "n_total": int(len(work)),
        "n_train": int(len(train_records)),
        "n_val": int(len(val_records)),
        "n_test": int(len(test_records)),
        "categories": categories,
    }
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print(f"[OK] train: {train_path}")
    print(f"[OK] val: {val_path}")
    print(f"[OK] test: {test_path}")
    print(f"[OK] metadata: {meta_path}")


if __name__ == "__main__":
    main()
