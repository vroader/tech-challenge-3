import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from local_model import resolve_model_source
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Voce e uma classificadora especializada em narrativas de violencia contra a mulher. "
    "Classifique o texto em exatamente uma categoria da lista. "
    "Nao invente categoria. "
    "Responda apenas com JSON valido no formato: "
    "{\"categoria\":\"NOME_EXATO_DA_CATEGORIA\",\"confianca\":0.00}."
)


def load_categories(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    cats = data.get("categories", [])
    if not cats:
        raise ValueError("Arquivo de categorias sem campo 'categories'.")
    return cats


def build_prompt(text: str, categories: List[str]) -> str:
    cat_block = "\n".join(categories)
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nCategorias permitidas:\n{cat_block}\n\n"
        f"Texto para classificar:\n{text}\n"
        "<|assistant|>\n"
    )


def parse_json_answer(generated: str, categories: List[str]) -> dict:
    start = generated.find("{")
    end = generated.rfind("}")
    if start >= 0 and end > start:
        candidate = generated[start : end + 1]
        try:
            parsed = json.loads(candidate)
            category = parsed.get("categoria", "")
            if category in categories:
                return parsed
        except json.JSONDecodeError:
            pass

    # fallback simples
    for cat in categories:
        if cat in generated:
            return {"categoria": cat, "confianca": 0.0}

    return {"categoria": "", "confianca": 0.0, "raw": generated}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia local com adapter LoRA treinado.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-dir", default="models/qwen2.5-7b-vitima-only-lora")
    parser.add_argument("--categories-json", default="data/processed/finetune/llm_local/categories.json")
    parser.add_argument("--text", required=True, help="Texto para classificar.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", default=None, help="Diretorio de cache para arquivos Hugging Face.")
    parser.add_argument("--allow-remote", action="store_true", help="Permite acesso ao Hugging Face Hub.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.allow_remote or os.getenv("HF_HUB_OFFLINE", "0") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    categories = load_categories(Path(args.categories_json))
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
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_remote,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    prompt = build_prompt(args.text, categories)
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

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = parse_json_answer(generated, categories)
    print(json.dumps(answer, ensure_ascii=False))


if __name__ == "__main__":
    main()
