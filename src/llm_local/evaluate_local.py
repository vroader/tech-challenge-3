import argparse
import json
import os
from pathlib import Path
from typing import List

from sklearn.metrics import accuracy_score, classification_report, f1_score

from local_model import resolve_model_source
from predict_local import load_categories, parse_json_answer, build_prompt  # noqa: E402

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliacao local do classificador LLM fine-tuned.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-dir", default="models/qwen2.5-7b-vitima-only-lora")
    parser.add_argument("--categories-json", default="data/processed/finetune/llm_local/categories.json")
    parser.add_argument("--test-jsonl", default="data/processed/finetune/llm_local/test.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Limita a avaliacao para teste rapido.")
    parser.add_argument("--cache-dir", default=None, help="Diretorio de cache para arquivos Hugging Face.")
    parser.add_argument("--allow-remote", action="store_true", help="Permite acesso ao Hugging Face Hub.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.allow_remote or os.getenv("HF_HUB_OFFLINE", "0") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    categories = load_categories(Path(args.categories_json))
    test_records = read_jsonl(Path(args.test_jsonl))
    if args.limit > 0:
        test_records = test_records[: args.limit]
    base_model_source = resolve_model_source(args.base_model, cache_dir=args.cache_dir, allow_remote=args.allow_remote)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_source,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_source,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_remote,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    y_true: List[str] = []
    y_pred: List[str] = []

    for row in test_records:
        text = str(row["input_text"])
        label = str(row["label"])

        prompt = build_prompt(text, categories)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        parsed = parse_json_answer(generated, categories)
        pred = parsed.get("categoria", "")

        y_true.append(label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"[METRICA] accuracy={acc:.4f}")
    print(f"[METRICA] macro_f1={macro_f1:.4f}")
    print("[METRICA] classificacao por classe:")
    print(report)


if __name__ == "__main__":
    main()
