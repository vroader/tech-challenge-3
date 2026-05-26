import argparse
import json
import logging
from pathlib import Path
from typing import List

import faiss
import joblib
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:  # pragma: no cover
        RecursiveCharacterTextSplitter = None

from langchain_openai import OpenAIEmbeddings


class LocalRecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs: List[Document]) -> List[Document]:
        out: List[Document] = []
        for doc in docs:
            text = doc.page_content or ""
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                out.append(Document(page_content=text[start:end], metadata=dict(doc.metadata)))
                if end >= len(text):
                    break
                start = max(0, end - self.chunk_overlap)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera indice vetorial FAISS para consulta RAG.")
    parser.add_argument("--input-file", default="data/processed/rag/base_rag_limpa.txt")
    parser.add_argument("--input-dir", default="docs/Referencial_rag", help="Usado se input-file nao existir.")
    parser.add_argument("--index-dir", default="data/processed/rag/faiss_index")
    parser.add_argument("--embedding-backend", default="openai", choices=["tfidf", "openai"])
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def read_input(path: Path) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(path)})]


def read_input_dir(path: Path) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {path}")
    docs: List[Document] = []
    for txt in sorted(path.rglob("*.txt")):
        if not txt.is_file():
            continue
        try:
            text = txt.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = txt.read_text(encoding="latin-1")
        docs.append(Document(page_content=text, metadata={"source": str(txt)}))
    if not docs:
        raise FileNotFoundError(f"Nenhum arquivo TXT encontrado em: {path}")
    return docs


def split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    else:
        logging.warning("RecursiveCharacterTextSplitter indisponivel; usando fallback local por caractere.")
        splitter = LocalRecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = splitter.split_documents(docs)
    for idx, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = idx
    return chunks


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    input_file = Path(args.input_file)
    index_dir = Path(args.index_dir)

    if input_file.exists():
        docs = read_input(input_file)
        input_desc = str(input_file)
    else:
        docs = read_input_dir(Path(args.input_dir))
        input_desc = str(Path(args.input_dir))

    chunks = split_docs(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    logging.info("Chunks para indexacao: %s", len(chunks))

    texts = [d.page_content for d in chunks]
    if args.embedding_backend == "openai":
        embeddings = OpenAIEmbeddings(model=args.embedding_model)
        vectors = embeddings.embed_documents(texts)
        matrix = np.array(vectors, dtype="float32")
    else:
        vectorizer = TfidfVectorizer(max_features=50000)
        matrix = vectorizer.fit_transform(texts).astype(np.float32).toarray()

    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    records = []
    for doc in chunks:
        records.append(
            {
                "source": doc.metadata.get("source", "desconhecido"),
                "chunk_id": doc.metadata.get("chunk_id", -1),
                "text": doc.page_content,
            }
        )
    with (index_dir / "documents.jsonl").open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.embedding_backend == "tfidf":
        joblib.dump(vectorizer, index_dir / "tfidf_vectorizer.joblib")

    manifest = {
        "input_file": input_desc,
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "chunks": len(chunks),
        "index_dir": str(index_dir),
    }
    with (index_dir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    logging.info("Indice FAISS salvo em: %s", index_dir)


if __name__ == "__main__":
    main()
