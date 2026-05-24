from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_model_source(model_id: str, cache_dir: str | None = None, allow_remote: bool = False) -> str:
    model_path = Path(model_id)
    if model_path.exists():
        return str(model_path)

    if allow_remote:
        return model_id

    attempts = []
    for current_cache_dir in (cache_dir, None):
        try:
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=current_cache_dir,
                local_files_only=True,
            )
            return local_path
        except Exception as exc:
            attempts.append((current_cache_dir, exc))

    cache_hint = cache_dir or "cache padrao do Hugging Face"
    raise RuntimeError(
        f"Nao foi possivel localizar o modelo '{model_id}' em disco. Procurei em '{cache_hint}' e no cache padrao do Hugging Face. "
        f"Baixe o modelo uma vez com internet ou aponte --base-model para uma pasta local contendo os arquivos do modelo."
    )