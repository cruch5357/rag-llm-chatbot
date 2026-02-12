import os
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


PERSIST_DIR = "storage/chroma"
COLLECTION_NAME = "docs"


def build_settings():
    # Embeddings locales (sin API key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # LLM local vía Ollama
    Settings.llm = Ollama(model="llama3.1", request_timeout=120)


def get_index():
    build_settings()

    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Carga el índice desde el vector store persistente
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
