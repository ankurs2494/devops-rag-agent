import chromadb
import requests
import hashlib
from pypdf import PdfReader
from tqdm import tqdm
from chromadb.config import Settings
import re

# ---------------- CONFIG ----------------

PDF_DIR = "./docs/"
CHROMA_DIR = "./chroma"
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"  # or "bge-m3"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

COLLECTION_MAP = {
    "kubernetes-networking.pdf": "kubernetes-networking",
    "k8s-storage.pdf": "k8s-storage",
    "terraform-aws.pdf": "terraform-aws",
    "terraform-azure.pdf": "terraform-azure",
    "ci-github-actions.pdf": "github-actions",
    "ollama_chromadb_ingestion_explained.pdf": "rag-architecture",
    "vector_database_guide.pdf": "vector-database",
}

# ----------------------------------------


def ollama_embed(text: str):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]




def chunk_text(text: str):
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < 3500:  # ~750 tokens
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Add overlap
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            overlap = chunks[i - 1][-600:]  # ~120 tokens
            final_chunks.append(overlap + chunk)

    return final_chunks


def stable_id(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pdf(path: str):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))
    return pages


def main():

    client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
    )

    for pdf_name, collection_name in COLLECTION_MAP.items():
        print(f"\n Ingesting {pdf_name} → {collection_name}")

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"domain": collection_name},
        )

        pdf_path = f"{PDF_DIR}/{pdf_name}"
        pages = load_pdf(pdf_path)

        for page_num, page_text in tqdm(pages):
            chunks = chunk_text(page_text)

            for idx, chunk in enumerate(chunks):
                doc_id = stable_id(f"{pdf_name}-{page_num}-{idx}-{chunk}")

                existing = collection.get(ids=[doc_id])
                if existing["ids"]:
                    continue

                embedding = ollama_embed(chunk)

                collection.add(
                    ids=[doc_id],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": pdf_name,
                        "page": page_num,
                        "collection": collection_name,
                    }],
                )

    print("\n Ingestion complete. ChromaDB is ready.")


if __name__ == "__main__":
    main()