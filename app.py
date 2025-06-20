import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from qdrant_client import QdrantClient

load_dotenv(".env")

st.header("Surya Tani ChatBot")

# Connect ke Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY", None)  # untuk cloud, kalau local bisa None
)

# Inisialisasi vectorstore dari koleksi yang sudah ada
embeddings = OpenAIEmbeddings()
vectorstore = Qdrant(
    client=client,
    collection_name="suryatani_collection",
    embeddings=embeddings
)

# Input pertanyaan
query = st.text_input("Tanya Produk Surya Tani")
if query:
    docs = vectorstore.similarity_search(query, k=3)

    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)

    st.write(response)
