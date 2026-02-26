# Ollama DevOps Agent

An **agentic DevOps assistant powered by a local LLM and Retrieval-Augmented Generation (RAG).**

Designed **to explore and understand the internal workings of Ollama, vector databases, and agentic DevOps systems.**

This project serves as a **learning and experimentation platform** to evaluate:

- How local LLMs behave in real DevOps scenarios
- How RAG with vector databases grounds agent knowledge
- How effectively a DevOps agent understands infrastructure, configs, logs, and CI/CD pipelines

The focus is on **hands-on experimentation and knowledge validation**, rather than polished chat demos or production automation.

---

##  Key Features

*  **Agentic DevOps Reasoning**
  Multi-step reasoning for debugging, incident analysis, and operational tasks.

*  **RAG (Retrieval-Augmented Generation)**
  Grounded, source-backed answers using:

  * Kubernetes **networking** (CNI, Services, Ingress, NetworkPolicies)
  * Kubernetes **storage** (PV, PVC, CSI, StorageClasses)
  * **Terraform** (AWS & Azure infrastructure)
  * **GitHub Actions** CI pipelines
  * **Vector database**–backed knowledge retrieval

*  **Fully Local & Private**
  Runs on **Ollama + llama.cpp** (Apple Silicon / Metal supported). No cloud dependency.


---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                DevOps User                                   │
│                     (question, error logs, timeouts, etc.)                   │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CLI / App Entry (main loop)                           │
│     - Reads user input                                                        │
│     - Sends query to agent build_chain(query)                                 │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Agent Orchestrator / Factory                          │
│                               get_agent()                                     │
│  Creates:                                                                      │
│   - Ollama LLM (llama3)                                                        │
│   - Ollama Embeddings (nomic-embed-text)                                       │
│   - ChatPromptTemplate (guardrails + formats)                                  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Query Router (resolve_collection)                    │
│  Input : query string                                                         │
│  Logic : keyword match scoring across domains                                  │
│                                                                              │
│   Collections:                                                                 │
│    - kubernetes-networking  (cni, calico, ingress, networkpolicy, dns...)     │
│    - k8s-storage            (pv, pvc, csi, storageclass...)                   │
│    - terraform              (terraform, provider, state...)                   │
│                                                                              │
│  Output: selected collection_name (string)                                     │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Vector Store Loader (Chroma Client)                         │
│   Chroma(                                                                     │
│     persist_directory="./chroma",                                             │
│     embedding_function=OllamaEmbeddings("nomic-embed-text"),                   │
│     collection_name=<routed_collection>                                       │
│   )                                                                            │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Retriever Builder (get_retriever)                     │
│   db.as_retriever(                                                            │
│     search_type="similarity",                                                 │
│     search_kwargs={"k": 5}                                                    │
│   )                                                                            │
│                                                                              │
│  Output: retriever                                                            │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Retrieval Step (RAG)                                │
│   retriever(query) → returns Top-K relevant docs/snippets                      │
│                                                                              │
│  Output: context (documents)                                                  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Prompt Assembly + Guardrails Layer                        │
│   ChatPromptTemplate injects:                                                  │
│    - "Context: {context}"                                                     │
│    - "Question: {question}"                                                   │
│                                                                              │
│   Built-in behavior (from your prompt):                                       │
│    - Intent detection (errors/logs → Troubleshooting format)                  │
│    - Commands allowed ONLY in troubleshooting                                 │
│    - Commands must be in ```bash blocks                                       │
│    - If missing context → say "I don't know..."                               │
│    - "No diagnostic commands found..." fallback                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Local LLM Inference (Ollama)                        │
│                     OllamaLLM(model="llama3", temp=0.7)                       │
│                                                                              │
│  Output: final response in strict Markdown format                             │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                Output to User                                │
│     - Markdown answer                                                         
│     - Troubleshooting or Concept format                                      │
│     - Commands only when allowed                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

* **LLM Runtime**: Ollama (llama.cpp backend)
* **Models**: nomic-embed-text
* **RAG**:
  * Embeddings via `/api/embed`
  * Vector DB (Chroma)
* **Inference Backend**: Metal (Apple Silicon)
* **Language**: Python
* **API Framework**: FastAPI
* **Web Interface**: HTML (lightweight UI for interacting with the agent)

---


It just act as a general-purpose chatbot

The agent is **grounded, auditable, and constrained** by RAG and tools.

---

## Current Status

* [x] Local LLM inference via Ollama
* [x] RAG pipeline (embeddings + retrieval)
* [ ] Tool execution sandboxing
* [ ] Multi-agent concurrency
* [ ] Persistent memory layer


## Agent Snippet

<img width="1261" height="586" alt="image" src="https://github.com/user-attachments/assets/24cbbe12-259a-4e30-9ed7-3db73d811d60" />

---

<img width="1252" height="580" alt="image" src="https://github.com/user-attachments/assets/bbfb65f8-fd42-4be5-8914-526c9cda8b81" />

---

## etcd Quorum Loss (kubernetes issue)

<img width="1879" height="828" alt="image" src="https://github.com/user-attachments/assets/b77af775-ac91-465e-8853-c2d65d75f076" />

---
<img width="1894" height="824" alt="image" src="https://github.com/user-attachments/assets/f61965a2-9b8b-42d3-a6a5-9ef63b657e7d" />

---
## Service Not Reachable (kubernetes issue)

<img width="1897" height="824" alt="image" src="https://github.com/user-attachments/assets/b0f76565-4e0e-414d-8a5d-427fd7f06125" />

---

<img width="1894" height="827" alt="image" src="https://github.com/user-attachments/assets/3106d716-09f1-466c-b230-f55785eb689f" />

