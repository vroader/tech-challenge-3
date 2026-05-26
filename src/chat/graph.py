from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .classifier_node import classify_text
from .friend_node import get_friend_response
from .mongo_logger import log_turn
from .rag_node import get_rag_response


class GraphState(TypedDict):
    messages: List[Dict[str, str]]  # [{role: "user"|"assistant", content: str}, ...]
    user_text: str
    categoria: str
    confianca: float
    session_id: str
    response: str
    branch: str


def build_graph(
    model,
    tokenizer,
    categories,
    faiss_index,
    docs_metadata,
    embeddings,
):
    """Build and compile the LangGraph StateGraph for the support chat.

    The graph is intended to be compiled once at app startup and invoked
    on every new user message.

    Flow:
        classify → (conditional) friend_node  → log → END
                               └─ rag_node   ↗
    """

    # ── Nodes ────────────────────────────────────────────────────────────────

    def classify_node(state: GraphState) -> GraphState:
        result = classify_text(state["user_text"], model, tokenizer, categories)  # model may be None (api-only)
        return {
            **state,
            "categoria": result.get("categoria", ""),
            "confianca": result.get("confianca", 0.0),
        }

    def friend_node(state: GraphState) -> GraphState:
        # Pass history WITHOUT the current user message as context
        response = get_friend_response(
            user_text=state["user_text"],
            history=state["messages"],
        )
        return {**state, "response": response, "branch": "friend"}

    def rag_node(state: GraphState) -> GraphState:
        response = get_rag_response(
            user_text=state["user_text"],
            categoria=state["categoria"],
            index=faiss_index,
            docs_metadata=docs_metadata,
            embeddings=embeddings,
        )
        return {**state, "response": response, "branch": "rag"}

    def log_node(state: GraphState) -> GraphState:
        log_turn(
            conversation_id=state["session_id"],
            user_text=state["user_text"],
            categoria_detectada=state["categoria"],
            confianca=state["confianca"],
            response_text=state["response"],
            branch=state["branch"],
        )
        return state

    # ── Routing ──────────────────────────────────────────────────────────────

    def route(state: GraphState) -> str:
        """Route to friend when no violence detected, otherwise to RAG advisory."""
        categoria = state.get("categoria", "")
        if not categoria or categoria == "SEM_VIOLÊNCIA":
            return "friend"
        return "rag"

    # ── Graph assembly ───────────────────────────────────────────────────────

    graph = StateGraph(GraphState)

    graph.add_node("classify", classify_node)
    graph.add_node("friend", friend_node)
    graph.add_node("rag", rag_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route,
        {"friend": "friend", "rag": "rag"},
    )

    graph.add_edge("friend", "log")
    graph.add_edge("rag", "log")
    graph.add_edge("log", END)

    return graph.compile()
