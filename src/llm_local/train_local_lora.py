import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from local_model import resolve_model_source
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino local LoRA/QLoRA para classificacao por categoria.")
    parser.add_argument("--train-jsonl", default="data/processed/finetune/llm_local/train.jsonl")
    parser.add_argument("--val-jsonl", default="data/processed/finetune/llm_local/val.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="models/qwen2.5-7b-vitima-only-lora")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None, help="Diretorio de cache para arquivos Hugging Face.")
    parser.add_argument("--allow-remote", action="store_true", help="Permite acesso ao Hugging Face Hub.")
    parser.add_argument("--use-4bit", action="store_true", help="Ativa QLoRA em GPU CUDA com bitsandbytes.")
    return parser.parse_args()


def tokenize_batch(batch: Dict[str, list], tokenizer, max_length: int) -> Dict[str, list]:
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
    return encoded


def build_model(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_source = resolve_model_source(args.base_model, cache_dir=args.cache_dir, allow_remote=args.allow_remote)

    quantization_config = None
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": args.cache_dir,
        "local_files_only": not args.allow_remote,
    }

    if args.use_4bit:
        if device != "cuda":
            raise RuntimeError("--use-4bit exige GPU CUDA.")
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("BitsAndBytesConfig indisponivel. Instale bitsandbytes.") from exc

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["dtype"] = torch.bfloat16
    elif device == "cuda":

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_source, **model_kwargs)
    except Exception as exc:
        if isinstance(exc, OSError) or "getaddrinfo failed" in str(exc).lower():
            raise RuntimeError(
                "Falha ao carregar o modelo base. Sem conectividade com Hugging Face e sem cache local disponivel. "
                "Opcoes: (1) rode com internet/proxy configurado; "
                "(2) pre-baixe o modelo e use --base-model apontando para o diretorio local; "
                "(3) rode com --local-files-only apos baixar o modelo em cache."
            ) from exc
        raise

    if device == "cuda" and not args.use_4bit:
        model = model.to("cuda")

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bia        model_kwargs["dtype"] = torch.bfloat16
s="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_cfg)
    return model


def main() -> None:
    args = parse_args()
    offline_enabled = not args.allow_remote or os.getenv("HF_HUB_OFFLINE", "0") == "1"
    if offline_enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {val_path}")

    try:
        base_model_source = resolve_model_source(args.base_model, cache_dir=args.cache_dir, allow_remote=args.allow_remote)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_source,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            local_files_only=not args.allow_remote,
        )
    except Exception as exc:
        if isinstance(exc, OSError) or "getaddrinfo failed" in str(exc).lower():
            raise RuntimeError(
                "Falha ao carregar o tokenizer. Sem conectividade com Hugging Face e sem cache local disponivel. "
                "Use um caminho local em --base-model, configure internet/proxy, ou rode com --local-files-only "
                "quando o modelo ja estiver em cache."
            ) from exc
        raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )

    tokenized = raw.map(
        lambda batch: tokenize_batch(batch, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    model = build_model(args)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=20,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    run_cfg = vars(args)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as fp:
        json.dump(run_cfg, fp, ensure_ascii=False, indent=2)

    print(f"[OK] adapter salvo em: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[ERRO] {exc}")
        raise SystemExit(1)
