import sys
from pathlib import Path

import faiss
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Expose src/rag so query_rag helpers can be imported
_RAG_DIR = Path(__file__).parent.parent / "rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from query_rag import format_docs, load_docs_metadata, retrieve_docs  # noqa: E402

load_dotenv()

RAG_SYSTEM_PROMPT = (
    "Você é uma assistente de acolhimento para mulheres. "
    "Use sempre linguagem gentil, sem julgamentos e com muito cuidado emocional. "
    "Nunca afirme categoricamente que a pessoa está sofrendo violência. "
    "Use expressões como 'o que você descreveu pode ser', 'talvez valha a pena buscar', "
    "'aparentemente há sinais que merecem atenção', 'existe a possibilidade de'. "
    "Oriente sobre medidas protetivas, canais de ajuda e direitos legais "
    "usando apenas o contexto recuperado. "
    "Não invente telefones, endereços, órgãos ou procedimentos."
)

RAG_HUMAN_TEMPLATE = (
    "A usuária compartilhou o seguinte relato:\n{user_text}\n\n"
    "O sistema identificou que o relato pode conter sinais de: {categoria}\n\n"
    "Contexto de referência legal e de acolhimento:\n{context}\n\n"
    "Responda de forma empática e suave, estruturando assim:\n"
    "1) Acolhimento e validação emocional\n"
    "2) O que esse tipo de situação pode representar (sem afirmar, apenas sugerir)\n"
    "3) Medidas protetivas e direitos que podem ser relevantes\n"
    "4) Canais de ajuda e onde buscar atendimento"
)


def load_rag_resources(
    index_dir: str = "data/processed/rag/faiss_index",
    embedding_model: str = "text-embedding-3-small",
):
    """Load FAISS index, document metadata and embeddings once at startup."""
    load_dotenv()
    index_path = Path(index_dir) / "index.faiss"
    docs_path = Path(index_dir) / "documents.jsonl"
    index = faiss.read_index(str(index_path))
    docs_metadata = load_docs_metadata(docs_path)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return index, docs_metadata, embeddings


def get_rag_response(
    user_text: str,
    categoria: str,
    index,
    docs_metadata,
    embeddings,
    top_k: int = 5,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    """Retrieve relevant context and generate a gentle advisory response."""
    # Build a query that combines the detected category with the user's text
    query = f"{categoria}: {user_text[:500]}"
    docs = retrieve_docs(query, embeddings, index, docs_metadata, top_k)
    context_text = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_TEMPLATE),
        ]
    )
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "user_text": user_text,
            "categoria": categoria,
            "context": context_text,
        }
    )
