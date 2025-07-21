import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from utils import load_documents, create_vectorstore

st.set_page_config(page_title="Chat with Docs", page_icon="📄")
st.title("📄 Chat with Your Documents")

# 📂 Upload documents
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("docs", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join("docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} file(s). Click below to process.")

# 🔄 Load and index documents
if st.button("📊 Index Documents"):
    with st.spinner("⬆️ Indexing documents..."):
        try:
            docs = load_documents("docs")
            if not docs:
                st.warning("No valid documents found in the 'docs' folder.")
            else:
                vectordb = create_vectorstore(docs)
                if vectordb:
                    if not st.secrets.get("GROQ_API_KEY"):
                        st.error("GROQ_API_KEY not found. Please add it to your Streamlit secrets.")
                        st.stop()
                    llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama3-70b-8192")
                    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
                    st.session_state.qa = qa
                    st.success("✅ Documents indexed and ready!")
                else:
                    st.warning("Could not extract any text from the documents. Please check the files.")
        except Exception as e:
            st.error(f"❌ Failed to index documents: {e}")

# 🔍 Chat with indexed content
if "qa" in st.session_state:
    query = st.text_input("Ask a question about your documents:")
    if query:
        with st.spinner("⏳ Thinking..."):
            try:
                response = st.session_state.qa.run(query)
                st.markdown(f"**Answer:** {response}")
            except Exception as e:
                st.error(f"⚠️ Failed to generate response: {e}")
else:
    st.info("⬆️ Upload and index your documents to begin.")
