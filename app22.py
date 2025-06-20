import streamlit as st
from PyPDF2 import PdfReader
import argparse
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv


load_dotenv('.env')


st.header("Surya Tani ChatBot")

pdf = st.file_uploader("Upload PDF Database", type="pdf")
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    #split menjadi chunk
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    #embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    pertanyaan = st.text_input("Tanya Produk Surya Tani")
    if pertanyaan:
        docs = knowledge_base.similarity_search(pertanyaan)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type='stuff')
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=pertanyaan)
        st.write(response)