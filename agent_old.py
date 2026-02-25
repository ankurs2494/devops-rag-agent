from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ---------------- CONFIG ----------------

CHROMA_DIR = "./chroma"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

COLLECTION_NAME = {
    "kubernetes-networking": {
        "keywords": [
            # CNI Plugins
            "cni", "calico", "cilium", "flannel", "weave", "ebpf", "bgp", "vxlan", "overlay",
            # Services
            "clusterip", "nodeport", "loadbalancer", "externalname", "kube-proxy", "ipvs", "iptables",
            # Ingress & Gateway
            "ingress", "gateway api", "layer 7", "l7", "http routing", "multi-tenancy", "traffic splitting",
            # Network Policy
            "networkpolicy", "podselector", "namespaceselector", "egress", "ingress rules",
            # DNS
            "coredns", "kube-dns", "stubdomains", "dns", "cluster.local",
            # Service Mesh
            "istio", "linkerd", "sidecar", "mutual tls", "mtls", "service mesh",
            # Core Concepts
            "pod ip", "nat", "flat network", "pod communication", "cross-node"
        ]
    },
    "k8s-storage": {
        "keywords": ["pv", "pvc", "csi", "storageclass", "volume"]
    },
    "terraform": {
        "keywords": ["terraform", "tf", "provider", "state"]
    },
}

DEFAULT_COLLECTION = "kubernetes-networking"  # fallback if no keywords match

# ----------------------------------------

def resolve_collection(query: str) -> str:
    """
    Routes the query to the correct ChromaDB collection
    by matching keywords. Falls back to DEFAULT_COLLECTION.
    """
    query_lower = query.lower()
    scores = {}

    for collection, config in COLLECTION_NAME.items():
        # Count how many keywords match in the query
        matches = sum(1 for kw in config["keywords"] if kw in query_lower)
        scores[collection] = matches

    best = max(scores, key=scores.get)

    # Only route to best if at least 1 keyword matched
    if scores[best] > 0:
        print(f"[Router] Matched collection: '{best}' (score: {scores[best]})")
        return best

    print(f"[Router] No match found, using default: '{DEFAULT_COLLECTION}'")
    return DEFAULT_COLLECTION



def get_retriever(query: str, embeddings):
    """
    Dynamically builds a retriever for the resolved collection.
    """
    collection = resolve_collection(query)

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=collection          # ✅ now always a string
    )

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )


def get_agent():
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.7,
        top_p=0.9
    )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert AI and DevOps assistant.

        Your role is to provide accurate, practical, real-world answers similar in tone to ChatGPT,
        while strictly grounding factual claims in the provided context.

        ────────────────────────────────────────
        INTENT DETECTION
        ────────────────────────────────────────

        If the question contains:
        - error messages
        - log snippets
        - words like "error", "failed", "timeout", "refused", "crash", "down", "not working", "x509"
        - or describes a system malfunction

        → Use TROUBLESHOOTING FORMAT.

        Otherwise → Use CONCEPT FORMAT.

        ────────────────────────────────────────
        CRITICAL RULES
        ────────────────────────────────────────
        - Use the provided context as the PRIMARY source of truth.
        - Do NOT invent tools, classifications, or examples.
        - Do NOT reclassify technologies unless explicitly stated in the context.
        - If the context does NOT contain enough information, say:
          "I don't know based on the provided documents."
        - You MAY use general knowledge ONLY to explain concepts already present in the context,
          but NOT to introduce new facts or examples.

        ────────────────────────────────────────
        COMMAND RULES (IMPORTANT)
        ────────────────────────────────────────

        - Commands are allowed ONLY in Troubleshooting Format.
        - ALL commands MUST be written inside triple-backtick bash blocks.
        - Example:
        ```bash
        kubectl get pods -A
        ```
        - NEVER return commands as objects, dicts, arrays, or JSON.
        - NEVER output [object Object].
        - NEVER use placeholder command text.
        - If no commands exist in context, write:
        "No diagnostic commands found in the provided documents."

        ────────────────────────────────────────
        TROUBLESHOOTING FORMAT
        ────────────────────────────────────────

        ## Problem Summary (Brief explanation of what is happening.)
        ## Likely Causes: Bullet list of common reasons for this issue.
        ## Symptoms Checklist
          - Observable behaviors
          - Log patterns
          - Errors
        ## Commands to Diagnose 
        ## Fix / Recovery Steps
        You MUST provide 3–8 numbered steps.
        Each step MUST include:
        - What to do (one sentence)
        - A command in a bash code block ONLY
        ```bash
        kubectl get networkpolicy
        ```
        - Use proper Markdown.
        - A short "Expected result" line with how should the command output should look if the issue is present.
        If Context does not contain commands for a step, write:
        "Command not found in the provided documents."
        
        ## Prevention (Optional)

        ────────────────────────────────────────
        CONCEPT FORMAT
        ────────────────────────────────────────

        ## What It Is (Clear, concise definition.)
        ## Why It Exists (Purpose and motivation.)
        ## How It Works (High-level explanation.)
        ## When to Use It (Real-world usage)
        ## Examples (Only if present in Context.)

        ────────────────────────────────────────
        FORMATTING RULES
        ────────────────────────────────────────

        - Use proper Markdown.
        - Use "-" for bullet points (one per line).
        - Do not use inline bullets like •.
        - Separate list items with blank lines.
        - Keep the response practical and concise.

        Context:
        {context}

        Question:
        {question}

        Answer (in Markdown):
        """
    )

    def build_chain(query: str):
        retriever = get_retriever(query, embeddings)
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        return chain

    return build_chain       # ✅ return the factory, not a single chain


def main():
    build_chain = get_agent()
    while True:
        q = input("Ask: ")
        if q.lower() in {"exit", "quit"}:
            break
        chain = build_chain(q)      # build chain with correct collection
        result = chain.invoke(q)
        print(result)

if __name__ == "__main__":
    main()