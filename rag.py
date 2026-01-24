from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

def load_docs():
    loader = PyPDFLoader("docs/sample.pdf")
    return loader.load()

def build_vectorstore():
    docs = load_docs()
    embeddings = OllamaEmbeddings(model="llama3")
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma")
    db.persist()
    print("Vector DB created.")

if __name__ == "__main__":
    build_vectorstore()