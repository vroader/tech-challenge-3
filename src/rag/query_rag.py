import argparse
import json
import logging
from pathlib import Path

import faiss
import joblib
import numpy as np
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

SYSTEM_PROMPT = (
    "Você é uma assistente de acolhimento para mulheres em situação de violência. "
    "Responda com linguagem clara, empática e prática, usando apenas o contexto recuperado. "
    "Priorize orientações de segurança imediata, canais oficiais de ajuda, medidas protetivas e direitos previstos na Lei Maria da Penha. "
    "Se o contexto não trouxer base suficiente para alguma parte, diga explicitamente que não encontrou essa informação no material de referência. "
    "Não invente telefones, endereços, órgãos ou procedimentos."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consulta RAG sobre orientacoes praticas para vitimas de violencia.")
    parser.add_argument("--index-dir", default="data/processed/rag/faiss_index")
    parser.add_argument("--question", required=True)
    parser.add_argument("--embedding-backend", default="openai", choices=["auto", "tfidf", "openai"])
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--answer-backend", default="openai", choices=["extractive", "openai"])
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def format_docs(docs) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        if isinstance(doc, dict):
            source = doc.get("source", "desconhecido")
            chunk_id = doc.get("chunk_id", "?")
            text = doc.get("page_content", "")
        else:
            source = doc.metadata.get("source", "desconhecido")
            chunk_id = doc.metadata.get("chunk_id", "?")
            text = doc.page_content
        blocks.append(f"[Fonte {i}] {source} (chunk {chunk_id})\n{text}")
    return "\n\n".join(blocks)


def load_docs_metadata(path: Path):
    docs = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def retrieve_docs(question: str, embeddings: OpenAIEmbeddings, index, docs_metadata, top_k: int):
    query_vec = np.array([embeddings.embed_query(question)], dtype="float32")
    faiss.normalize_L2(query_vec)
    _, ids = index.search(query_vec, top_k)

    docs = []
    for idx in ids[0].tolist():
        if idx < 0 or idx >= len(docs_metadata):
            continue
        rec = docs_metadata[idx]
        docs.append(
            {
                "source": rec.get("source", "desconhecido"),
                "chunk_id": rec.get("chunk_id", "?"),
                "page_content": rec.get("text", ""),
            }
        )
    return docs


def retrieve_docs_tfidf(question: str, vectorizer, index, docs_metadata, top_k: int):
    query_vec = vectorizer.transform([question]).astype(np.float32).toarray()
    faiss.normalize_L2(query_vec)
    _, ids = index.search(query_vec, top_k)

    docs = []
    for idx in ids[0].tolist():
        if idx < 0 or idx >= len(docs_metadata):
            continue
        rec = docs_metadata[idx]
        docs.append(
            {
                "source": rec.get("source", "desconhecido"),
                "chunk_id": rec.get("chunk_id", "?"),
                "page_content": rec.get("text", ""),
            }
        )
    return docs


def extractive_answer(question: str, docs) -> str:
    if not docs:
        return "Nao encontrei trechos relevantes na base para responder com seguranca."

    lines = [
        "1) O que fazer agora:",
        "- Busque imediatamente um local seguro e acione canais oficiais de ajuda quando houver risco iminente.",
        "",
        "2) Direitos e medidas protetivas cabiveis:",
        "- Com base nos trechos recuperados, verifique orientacoes sobre medidas protetivas e registro formal da ocorrencia.",
        "",
        "3) Canais de ajuda e onde procurar atendimento:",
    ]
    for i, doc in enumerate(docs[:5], start=1):
        snippet = " ".join(doc["page_content"].split())[:380]
        lines.append(f"- Trecho {i} ({doc['source']} | chunk {doc['chunk_id']}): {snippet}")
    lines.append("")
    lines.append(f"Pergunta original: {question}")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"Indice nao encontrado: {index_dir}")
    index_file = index_dir / "index.faiss"
    docs_file = index_dir / "documents.jsonl"
    if not index_file.exists() or not docs_file.exists():
        raise FileNotFoundError(
            "Indice incompleto. Esperado: index.faiss e documents.jsonl no diretorio informado."
        )

    index = faiss.read_index(str(index_file))
    docs_metadata = load_docs_metadata(docs_file)
    manifest = {}
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    embedding_backend = args.embedding_backend
    if embedding_backend == "auto":
        embedding_backend = manifest.get("embedding_backend", "tfidf")

    if embedding_backend == "openai":
        embeddings = OpenAIEmbeddings(model=args.embedding_model)
        retrieved_docs = retrieve_docs(
            question=args.question,
            embeddings=embeddings,
            index=index,
            docs_metadata=docs_metadata,
            top_k=args.top_k,
        )
    else:
        vectorizer_path = index_dir / "tfidf_vectorizer.joblib"
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                "Vectorizer TF-IDF nao encontrado. Reindexe com --embedding-backend tfidf."
            )
        vectorizer = joblib.load(vectorizer_path)
        retrieved_docs = retrieve_docs_tfidf(
            question=args.question,
            vectorizer=vectorizer,
            index=index,
            docs_metadata=docs_metadata,
            top_k=args.top_k,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Pergunta da usuaria:\n{question}\n\n"
                "Contexto de referencia:\n{context}\n\n"
                "Responda em portugues, em formato objetivo:\n"
                "1) O que fazer agora\n"
                "2) Direitos e medidas protetivas cabiveis\n"
                "3) Canais de ajuda e onde procurar atendimento",
            ),
        ]
    )

    if args.answer_backend == "openai":
        llm = ChatOpenAI(model=args.llm_model, temperature=args.temperature)
        context_text = format_docs(retrieved_docs)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": args.question})
    else:
        chain = RunnableLambda(lambda x: extractive_answer(x["question"], x["docs"]))
        answer = chain.invoke({"question": args.question, "docs": retrieved_docs})

    print(answer)

    if args.show_sources:
        print("\n--- FONTES RECUPERADAS ---")
        for i, doc in enumerate(retrieved_docs, start=1):
            print(f"{i}. {doc['source']} (chunk {doc['chunk_id']})")


if __name__ == "__main__":
    main()
