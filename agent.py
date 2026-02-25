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


# Configuration
CHROMA_DIR = "./chroma"
EMBED_MODEL = "nomic-embed-text"   # MUST match ingestion
LLM_MODEL = "llama3"
COLLECTION_NAME = {
    "k8s-networking": {
        "keywords": ["cni", "calico", "flannel", "cilium", "service", "ingress", "networkpolicy"]
    },
    "k8s-storage": {
        "keywords": ["pv", "pvc", "csi", "storageclass", "volume"]
    },
    "terraform": {
        "keywords": ["terraform", "tf", "provider", "state"]
    },
}

# Retrievers per collection
def build_retriever(COLLECTION_NAME, embeddings):

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    total_docs = db._collection.count()
    fetch_k = min(20, total_docs)
    k = min(5, total_docs)

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k
        }
    )

    # return db.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         "k": 5,
    #         "fetch_k": 20
    #     }
    # )



# router function (LLM-based)
def route_question(question: str, llm) -> str:
    ROUTER_PROMPT = """
    You are a routing assistant.

    Choose the best collection for the question.

    Collections:
    - k8s-networking
    - k8s-storage
    - terraform

    Question:
    {question}

    Respond with ONLY the collection name.
    """
    
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt | llm
    
    result = chain.invoke({"question": question})
    collection = result.strip().lower()
    
    # Validate the response
    if collection in COLLECTION_NAME:
        return collection
    return "k8s-networking"  # fallback



# get context function
def get_context(question, retrievers, llm):
    collection = route_question(question, llm)
    docs = retrievers[collection].invoke(question)
    return docs


def get_agent():

    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0,
        top_p=1.0,
    )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    retrievers = {
    name: build_retriever(name, embeddings)
    for name in COLLECTION_NAME.keys()
    }

    prompt = ChatPromptTemplate.from_template(
    """
    You are an expert AI and DevOps assistant.

    Your role is to provide accurate, practical, real-world answers similar in tone to ChatGPT,
    while strictly grounding factual claims in the provided context.

    SYSTEM:
    - All code MUST be inside Markdown triple backticks
    - If code exists outside a block, the answer is invalid
    - Prefer fenced code blocks over inline code

    CRITICAL RULES:
    - Use the provided context as the PRIMARY source of truth.
    - Include code snippets in code blocks.
    - Do NOT invent tools, classifications, or examples.
    - Do NOT reclassify technologies unless explicitly stated in the context.
    - If the context does NOT contain enough information, say:
      "I don't know based on the provided documents."
    - You MAY use general knowledge ONLY to explain concepts already present in the context,
      but NOT to introduce new facts or examples.

    Response style:
    - Practical and example-driven
    - technically accurate
    - Use headings and bullet points where helpful
    - Use code format while giving commands or code snippets

    For technical topics, structure the answer as:
    1. What it is
    2. Why it exists
    3. When to use it
    4. Symptoms checklist (include command ONLY if the question is about troubleshooting)
    5. Commands 
    6. Examples

    Formatting rules:
    - Use proper Markdown
    - Use code format while giving commands or code snippets
    - Use emojis sparingly to enhance clarity
    - Use "-" for bullet points (one per line)
    - Do NOT use inline bullets like •
    - Separate list items with new lines

    FINAL REMINDER:
    - If any command or code is present, it MUST be inside a Markdown code block.
    - Responses violating this are incorrect.

    Examples of code formatting:
    Incorrect:
    kubectl get pods

    Correct:
    ```bash
    kubectl get pods

    Context:
    {context}

    Question:
    {question}

    Answer (in Markdown):
    """
)


    chain = (
        {
            "context": lambda q: get_context(q, retrievers, llm),
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
        try:
            result = agent.invoke(q)
            print(result)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()