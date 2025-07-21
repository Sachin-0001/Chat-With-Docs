import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


def load_documents(doc_folder):
    docs = []
    for file in os.listdir(doc_folder):
        path = os.path.join(doc_folder, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs


def create_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Extract page_content, convert to string, and filter out empty or whitespace-only strings.
    texts = [str(doc.page_content).strip() for doc in splits]
    texts = [text for text in texts if text]

    if not texts:
        return None

    embeddings = load_embeddings()

    # Use FAISS.from_texts since we have a clean list of strings.
    vectordb = FAISS.from_texts(texts=texts, embedding=embeddings)
    return vectordb

