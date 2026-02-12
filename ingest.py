import os
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


DATA_DIR = "data"
PERSIST_DIR = "storage/chroma"
COLLECTION_NAME = "docs"


def main():
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        raise SystemExit("❌ No hay documentos en /data. Agrega PDFs/TXT/MD y reintenta.")

    # Embeddings (mismo que rag.py)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()

    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Esto crea embeddings + guarda en Chroma
    _ = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    print("✅ Ingesta completada. Vector DB persistido en:", PERSIST_DIR)


if __name__ == "__main__":
    main()
