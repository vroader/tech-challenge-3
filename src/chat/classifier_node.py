import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Expose src/llm_local so predict_local and local_model can be imported
_LLM_LOCAL_DIR = Path(__file__).parent.parent / "llm_local"
if str(_LLM_LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_LOCAL_DIR))

from local_model import resolve_model_source  # noqa: E402
from peft import PeftModel  # noqa: E402
from predict_local import build_prompt, load_categories, parse_json_answer  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# API-only mode helpers (used when USE_LOCAL_CLASSIFIER=false)
# ---------------------------------------------------------------------------

_GPT_CLASSIFY_SYSTEM = (
    "Você é uma classificadora especializada em narrativas de violência contra a mulher. "
    "Classifique o texto em exatamente uma categoria da lista fornecida. "
    "Não invente categoria. "
    'Responda apenas com JSON válido no formato: {"categoria":"NOME_EXATO","confianca":0.00}.'
)


def _load_categories_from_json(
    categories_json: str = "data/processed/finetune/llm_local/categories.json",
) -> List[str]:
    path = Path(categories_json)
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data.get("categories", [])


def classify_with_gpt(
    text: str,
    categories: List[str],
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """Classify text using OpenAI API when the local model is unavailable."""
    from langchain_openai import ChatOpenAI  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    cat_block = "\n".join(categories)
    llm = ChatOpenAI(model=llm_model, temperature=0)
    messages = [
        SystemMessage(content=_GPT_CLASSIFY_SYSTEM),
        HumanMessage(
            content=f"Categorias permitidas:\n{cat_block}\n\nTexto para classificar:\n{text}"
        ),
    ]
    response = llm.invoke(messages)
    raw = response.content
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if parsed.get("categoria") in categories:
                return parsed
        except json.JSONDecodeError:
            pass
    for cat in categories:
        if cat in raw:
            return {"categoria": cat, "confianca": 0.0}
    return {"categoria": "", "confianca": 0.0}


def load_classifier(
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_dir: str = "models/qwen2.5-7b-vitima-only-lora",
    categories_json: str = "data/processed/finetune/llm_local/categories.json",
    cache_dir=None,
    allow_remote: bool = False,
    api_only: bool = False,
) -> tuple:
    """Load model, tokenizer and categories once at application startup.

    When *api_only* is True (or the env var USE_LOCAL_CLASSIFIER=false is set)
    the function skips loading the local GPU model and returns (None, None, categories)
    so the graph falls back to classify_with_gpt().
    """
    if api_only or os.getenv("USE_LOCAL_CLASSIFIER", "true").lower() == "false":
        categories = _load_categories_from_json(categories_json)
        return None, None, categories
    if not allow_remote:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    categories = load_categories(Path(categories_json))
    base_model_source = resolve_model_source(
        base_model_id, cache_dir=cache_dir, allow_remote=allow_remote
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_source,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=not allow_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_source,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        cache_dir=cache_dir,
        local_files_only=not allow_remote,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    return model, tokenizer, categories


def classify_text(
    text: str,
    model: Optional[object],
    tokenizer: Optional[object],
    categories: List[str],
    max_new_tokens: int = 128,
) -> dict:
    """Run a single classification inference.

    When *model* is None (API-only mode) delegates to classify_with_gpt().
    Returns dict with keys ``categoria`` (str) and ``confianca`` (float).
    """
    if model is None:
        return classify_with_gpt(text, categories)
    prompt = build_prompt(text, categories)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return parse_json_answer(generated, categories)
