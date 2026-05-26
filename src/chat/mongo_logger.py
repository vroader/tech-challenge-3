import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        from pymongo import MongoClient

        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        client.server_info()  # triggers the actual connection attempt
        _collection = client["tech_challenge"]["conversations"]
        logger.info("MongoDB conectado: %s", mongo_uri)
    except Exception as exc:
        logger.warning("MongoDB indisponível — logs de conversa serão ignorados: %s", exc)
        _collection = None
    return _collection


def log_turn(
    conversation_id: str,
    user_text: str,
    categoria_detectada: str,
    confianca: float,
    response_text: str,
    branch: str,
) -> None:
    """Insert one conversation turn into MongoDB.

    Fails silently: if MongoDB is unavailable the chat keeps working normally.
    Schema:
        conversation_id  str    – shared by all turns in the same session
        timestamp        datetime UTC
        user_text        str
        categoria_detectada str
        confianca        float
        response_text    str
        branch           "friend" | "rag"
    """
    collection = _get_collection()
    if collection is None:
        return
    try:
        collection.insert_one(
            {
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc),
                "user_text": user_text,
                "categoria_detectada": categoria_detectada,
                "confianca": float(confianca),
                "response_text": response_text,
                "branch": branch,
            }
        )
    except Exception as exc:
        logger.warning("Erro ao salvar turno no MongoDB: %s", exc)
