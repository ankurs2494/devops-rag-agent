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
┌─────────────┐
│ DevOps User │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Agent Orchestrator│
└──────┬───────────┘
       │
       |
       │
       ├─► RAG Retriever
       │     ├─ Vector DB
       │     └─ Document Index
       │
       ▼
┌──────────────────┐
│ Local LLM        |
|   (Ollama)       │                
└──────────────────┘
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


