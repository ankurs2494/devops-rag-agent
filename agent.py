"""
What this script does:

- Uses Ollama for:
    LLM inference (llama3)
    Text embeddings (nomic-embed-text)

- Uses ChromaDB as a vector store

- Implements Retrieval-Augmented Generation (RAG):
    User asks a question
    Relevant documents are retrieved from Chroma
    Retrieved context + question are injected into a prompt
    LLM generates an answer grounded in that context
"""


from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ---------------- CONFIG ----------------

CHROMA_DIR = "./chroma"
EMBED_MODEL = "nomic-embed-text"   # MUST match ingestion
LLM_MODEL = "llama3"
COLLECTION_NAME = "k8s-networking"  # start with one

# ----------------------------------------

def get_agent():
    """
    Factory function that:
        Initializes models
        Builds retriever
        Creates the RAG chain
    """

    llm = OllamaLLM(
        model="llama3",
        temperature=0.7,
        top_p=0.9
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
        }
    )

    prompt = ChatPromptTemplate.from_template(
    """
    You are an expert AI and DevOps assistant.

    Your role is to provide accurate, practical, real-world answers similar in tone to ChatGPT,
    while strictly grounding factual claims in the provided context.

    CRITICAL RULES:
    - Use the provided context as the PRIMARY source of truth.
    - Do NOT invent tools, classifications, or examples.
    - Do NOT reclassify technologies unless explicitly stated in the context.
    - If the context does NOT contain enough information, say:
      "I don't know based on the provided documents."
    - You MAY use general knowledge ONLY to explain concepts already present in the context,
      but NOT to introduce new facts or examples.

    Response style:
    - Practical and concise
    - Beginner-friendly but technically accurate
    - Use headings and bullet points where helpful
    - Explain concepts clearly, not academically

    For technical topics, structure the answer as:
    1. What it is
    2. Why it exists
    3. When to use it
    4. Examples (ONLY if present in the context)

    Formatting rules:
    - Use proper Markdown
    - Use "-" for bullet points (one per line)
    - Do NOT use inline bullets like •
    - Separate list items with new lines

    Context:
    {context}

    Question:
    {question}

    Answer (in Markdown):
    """
    )


    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return chain

def main():
    agent = get_agent()
    while True:
        q = input("Ask: ")
        if q.lower() in {"exit", "quit"}:
            break
        result = agent.invoke(q)
        print(result)

if __name__ == "__main__":
    main()