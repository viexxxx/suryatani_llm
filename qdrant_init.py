from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

collection_name = "suryatani_collection"

# Buat collection jika belum ada
if collection_name not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# Load dan embed PDF
loader = PyPDFLoader("data/prabowo.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=qdrant_url,
    api_key=qdrant_api_key,
    collection_name=collection_name,
)

print("[INFO] Embedding selesai dan disimpan ke Qdrant.")
