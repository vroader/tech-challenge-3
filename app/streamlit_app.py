import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ───────────────────────────────────────────────────────────────
# Make src/ importable so "from chat.xxx import ..." resolves correctly.
ROOT = Path(__file__).resolve().parent.parent
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from chat.classifier_node import load_classifier  # noqa: E402
from chat.graph import build_graph  # noqa: E402
from chat.rag_node import load_rag_resources  # noqa: E402

# ── Detect operating mode ─────────────────────────────────────────────────────
# Set USE_LOCAL_CLASSIFIER=false in .env to run without the local GPU model.
_API_ONLY = os.getenv("USE_LOCAL_CLASSIFIER", "true").lower() == "false"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Conversa de Apoio",
    page_icon="💜",
    layout="centered",
)

# ── Cached resource loaders (run once per Streamlit server process) ───────────


@st.cache_resource(
    show_spinner="Carregando modelo de análise..."
    if not _API_ONLY
    else "Configurando classificador via API..."
)
def _load_classifier():
    return load_classifier()


@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def _load_rag():
    return load_rag_resources()


# ── Session state initialisation ─────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None

if "graph" not in st.session_state:
    model, tokenizer, categories = _load_classifier()
    faiss_index, docs_metadata, embeddings = _load_rag()
    st.session_state.graph = build_graph(
        model, tokenizer, categories, faiss_index, docs_metadata, embeddings
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("💜 Apoio Emocional")
    st.caption("Um espaço seguro para conversar à vontade.")
    st.divider()
    if st.button("Nova conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.last_diagnosis = None
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("Como você está se sentindo?")
st.caption("Estou aqui para ouvir. Pode falar à vontade. 💬")

# Render all previous turns from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input — always anchored to the bottom of the page
user_input = st.chat_input("Escreva aqui...")

if user_input:
    # 1. Display user message immediately (inline, before adding to history)
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Process through LangGraph and display assistant response inline
    with st.chat_message("assistant"):
        with st.spinner(""):
            result = st.session_state.graph.invoke(
                {
                    # Pass history WITHOUT the current message so nodes have clean context
                    "messages": list(st.session_state.messages),
                    "user_text": user_input,
                    "categoria": "",
                    "confianca": 0.0,
                    "session_id": st.session_state.session_id,
                    "response": "",
                    "branch": "",
                }
            )
        response = result["response"]
        st.markdown(response)

    # 3. Persist both turns to history for the next render cycle
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 4. Store diagnosis so the sidebar panel can render it on this same rerun
    st.session_state.last_diagnosis = {
        "categoria": result.get("categoria", ""),
        "confianca": result.get("confianca", 0.0),
        "branch": result.get("branch", ""),
    }

# ── Sidebar: diagnostic panel (rendered after graph invoke so it shows current result) ────
with st.sidebar:
    st.divider()
    diag = st.session_state.last_diagnosis
    if diag:
        st.subheader("🔍 Diagnóstico IA")
        if _API_ONLY:
            st.caption("Classificação via API (modo sem modelo local). Exibido apenas para avaliação do protótipo.")
        else:
            st.caption("Resultado do modelo fine-tuned (Qwen2.5-7B + LoRA). Exibido apenas para avaliação do protótipo.")
        categoria = diag["categoria"] or "—"
        confianca = float(diag["confianca"])
        branch = diag["branch"]

        if categoria == "SEM_VIOLÊNCIA":
            st.success(f"✅ **{categoria}**")
        elif categoria and categoria != "—":
            st.error(f"⚠️ **{categoria}**")
        else:
            st.info("Aguardando análise…")

        st.caption("Confiança do classificador")
        st.progress(min(confianca, 1.0), text=f"{confianca:.0%}")

        if branch == "friend":
            st.caption("Caminho ativo: 👫 Conversa de apoio")
        elif branch == "rag":
            st.caption("Caminho ativo: ⚖️ Orientação jurídica")

        st.divider()

    with st.expander("ℹ️ Sobre os modelos"):
        if _API_ONLY:
            st.markdown(
                "🔁 **Modo API-only** ativo  \n"
                "Classificador local desabilitado.\n\n"
                "🤖 **Classificador** — OpenAI API  \n"
                "`gpt-4o-mini` (zero-shot)  \n"
                "Detecta a categoria do relato.\n\n"
                "💬 **Resposta** — OpenAI API  \n"
                "`gpt-4o-mini` + RAG  \n"
                "Gera a conversa de apoio ou a orientação jurídica com base em "
                "documentos sobre legislação e serviços de apoio à mulher."
            )
        else:
            st.markdown(
                "🧠 **Classificador** — local (GPU)  \n"
                "Qwen2.5-7B + LoRA (fine-tuning)  \n"
                "Detecta a categoria do relato.  \n"
                "_Exibido apenas para avaliação do protótipo._\n\n"
                "💬 **Resposta** — OpenAI API  \n"
                "`gpt-4o-mini` + RAG  \n"
                "Gera a conversa de apoio ou a orientação jurídica com base em "
                "documentos sobre legislação e serviços de apoio à mulher."
            )
    st.caption(
        "Este espaço é confidencial. As conversas são armazenadas de forma anonimizada "
        "para melhoria contínua do serviço."
    )
