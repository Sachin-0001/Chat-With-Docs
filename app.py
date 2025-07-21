import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from config import GROQ_API_KEY
from utils import load_documents, create_vectorstore

st.set_page_config(page_title="Chat with Docs", page_icon="📄")
st.title("📄 Chat with Your Documents")

# 📂 Upload documents
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    os.makedirs("docs", exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join("docs", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} file(s). Click below to process.")

# 🔄 Load and index documents
if st.button("📊 Index Documents") or "qa" not in st.session_state:
    with st.spinner("⬆️ Indexing documents..."):
        if not os.path.exists("docs"):
            os.makedirs("docs")
        docs = load_documents("docs")
        vectordb = create_vectorstore(docs)
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
        st.session_state.qa = qa
        st.success("✅ Documents indexed and ready!")

# 🔍 Chat with indexed content
query = st.text_input("Ask a question about your documents:")
if query:
    with st.spinner("⏳ Thinking..."):
        response = st.session_state.qa.run(query)
        st.markdown(f"**Answer:** {response}")
