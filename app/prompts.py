"""Shared prompt templates used by both LCEL chain and LangGraph graph."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXTUALIZE_SYSTEM = (
    "Given the chat history and the latest user question, "
    "rewrite the question so it is fully standalone — no references to prior conversation. "
    "Do NOT answer it; just reformulate if needed, otherwise return it as-is."
)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_SYSTEM = (
    "You are a helpful AI assistant that answers questions based on the provided context. "
    "Use the following retrieved documents to answer the user's question accurately.\n\n"
    "Rules:\n"
    "- Base your answer ONLY on the provided context.\n"
    "- If the context doesn't contain enough information, say: "
    '"I don\'t have enough information in my knowledge base to answer that question."\n'
    "- Cite your sources by mentioning the filename and page number, e.g. (source: report.pdf, page 3).\n"
    "- Be concise but thorough.\n\n"
    "Context:\n{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
