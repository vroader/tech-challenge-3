import argparse
import json
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - fallback for older LangChain layouts
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:  # pragma: no cover - fallback without splitter package
        RecursiveCharacterTextSplitter = None

try:
    from langchain_community.document_loaders import TextLoader  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    TextLoader = None


class LocalRecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_documents(self, docs: List[Document]) -> List[Document]:
        out: List[Document] = []
        for doc in docs:
            text = doc.page_content or ""
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk = text[start:end]
                out.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
                if end >= len(text):
                    break
                start = max(0, end - self.chunk_overlap)
        return out

SYSTEM_PROMPT = (
    "Você é uma inteligência artificial especialista em extrair orientações práticas de acolhimento. Leia o texto técnico "
    "fornecido e extraia APENAS as instruções, direitos, canais de ajuda, telefones, órgãos de atendimento ou ações diretas "
    "que uma mulher em situação de violência pode tomar por conta própria. Ignore completamente dados estatísticos, discussões "
    "acadêmicas, dotações orçamentárias, termos de gestão pública, capacitação de servidores ou fluxos de trabalho internos "
    "para funcionários públicos. Se o trecho não contiver NENHUMA orientação prática aplicável diretamente à vítima, responda "
    "estritamente com a palavra: 'REMOVER'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Processa arquivos TXT para criar uma base RAG sintetizada e focada em orientacao pratica a vitimas."
    )
    parser.add_argument("--input-dir", default="docs/Referencial_rag", help="Pasta com os arquivos TXT de referencia.")
    parser.add_argument("--output-file", default="data/processed/rag/base_rag_limpa.txt", help="Arquivo TXT consolidado de saida.")
    parser.add_argument(
        "--output-jsonl",
        default="data/processed/rag/base_rag_limpa.jsonl",
        help="Arquivo JSONL com rastreabilidade por chunk.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Modelo de chat a ser usado na sintetizacao.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--glob", default="*.txt", help="Padrao de busca dentro da pasta de entrada.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_documents(input_dir: Path, file_glob: str) -> List[Document]:
    docs: List[Document] = []
    paths = sorted(input_dir.rglob(file_glob))
    if not paths:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {input_dir} com padrao '{file_glob}'.")

    for path in paths:
        if not path.is_file():
            continue
        if TextLoader is not None:
            # Tenta utf-8 primeiro e fallback para latin-1 para documentos legados.
            try:
                loader = TextLoader(str(path), encoding="utf-8")
                loaded = loader.load()
            except UnicodeDecodeError:
                loader = TextLoader(str(path), encoding="latin-1")
                loaded = loader.load()
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            loaded = [Document(page_content=text, metadata={"source": str(path)})]

        for d in loaded:
            d.metadata["source"] = str(path)
            docs.append(d)

    return docs


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    else:
        logging.warning("RecursiveCharacterTextSplitter indisponivel; usando fallback local por caractere.")
        splitter = LocalRecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    chunks = splitter.split_documents(docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


def build_chain(model_name: str, temperature: float):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Arquivo: {source}\n"
                "ID do trecho: {chunk_id}\n"
                "Trecho tecnico para analise:\n{chunk_text}",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def is_remove_response(text: str) -> bool:
    return text.strip().upper() == "REMOVER"


def process_chunks(chain, chunks: List[Document], batch_size: int, max_concurrency: int) -> List[dict]:
    kept: List[dict] = []

    for start in range(0, len(chunks), batch_size):
        window = chunks[start : start + batch_size]
        payload = [
            {
                "source": item.metadata.get("source", "desconhecido"),
                "chunk_id": item.metadata.get("chunk_id", -1),
                "chunk_text": item.page_content,
            }
            for item in window
        ]

        try:
            responses = chain.batch(payload, config={"max_concurrency": max_concurrency})
        except Exception:
            # Fallback sequencial caso o provedor limite batch nesse momento.
            responses = [chain.invoke(p) for p in payload]

        for in_item, out_text in zip(payload, responses):
            normalized = out_text.strip()
            if is_remove_response(normalized):
                continue
            kept.append(
                {
                    "source": in_item["source"],
                    "chunk_id": in_item["chunk_id"],
                    "sintese": normalized,
                }
            )

        logging.info("Processados %s/%s chunks", min(start + batch_size, len(chunks)), len(chunks))

    return kept


def save_outputs(records: List[dict], output_txt: Path, output_jsonl: Path) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    merged_lines: List[str] = []
    with output_jsonl.open("w", encoding="utf-8") as fp_jsonl:
        for rec in records:
            fp_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            merged_lines.append(rec["sintese"])

    with output_txt.open("w", encoding="utf-8") as fp_txt:
        fp_txt.write("\n\n".join(merged_lines).strip() + "\n")


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta de entrada nao encontrada: {input_dir}")

    docs = load_documents(input_dir, args.glob)
    chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    logging.info("Arquivos carregados: %s", len(docs))
    logging.info("Total de chunks: %s", len(chunks))

    chain = build_chain(model_name=args.model, temperature=args.temperature)
    records = process_chunks(
        chain=chain,
        chunks=chunks,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
    )

    out_txt = Path(args.output_file)
    out_jsonl = Path(args.output_jsonl)
    save_outputs(records, output_txt=out_txt, output_jsonl=out_jsonl)

    logging.info("Trechos mantidos apos filtro: %s", len(records))
    logging.info("Saida TXT: %s", out_txt)
    logging.info("Saida JSONL: %s", out_jsonl)


if __name__ == "__main__":
    main()
