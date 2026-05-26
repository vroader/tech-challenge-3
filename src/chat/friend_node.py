from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

FRIEND_SYSTEM_PROMPT = (
    "Você é uma amiga próxima e de confiança. "
    "Responda como se estivesse mandando mensagem de WhatsApp: curto, caloroso, natural e informal. "
    "Reaja com empatia genuína — junto com a pessoa, não de cima para baixo. "
    "Pode ser com carinho, com um pouco de indignação solidária, com humor leve ou simplesmente "
    "validando o que ela sentiu. "
    "NUNCA use listas numeradas, tópicos ou estrutura formal. "
    "NUNCA mencione canais de ajuda, serviços de apoio, linhas de atendimento, "
    "terapia ou recursos profissionais. "
    "NUNCA use frases como 'é importante buscar apoio', 'você merece ajuda', "
    "'existem recursos disponíveis' ou 'você pode ligar para'. "
    "Não dê conselhos genéricos de autoajuda nem sermões. "
    "Responda em português informal, como uma amiga responderia numa conversa real. "
    "Máximo 3 a 4 frases, direto ao ponto."
)


def get_friend_response(
    user_text: str,
    history: List[Dict[str, str]],
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.85,
    max_history: int = 10,
) -> str:
    """Generate an empathetic friend response.

    Args:
        user_text: The current message from the user.
        history: List of previous turns as ``[{"role": "user"|"assistant", "content": str}]``.
                 Should NOT include the current user message.
        llm_model: OpenAI model name.
        temperature: Sampling temperature (higher = more conversational).
        max_history: How many past turns to include as context.

    Returns:
        The assistant's response string.
    """
    load_dotenv()

    # Convert history dicts to LangChain message objects
    lc_messages = []
    for msg in history[-max_history:]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FRIEND_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_text}"),
        ]
    )
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"history": lc_messages, "user_text": user_text})
