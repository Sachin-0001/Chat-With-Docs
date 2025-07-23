import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from utils import load_documents, create_vectorstore
from langchain.document_loaders import PyPDFLoader, TextLoader
import tempfile

st.set_page_config(page_title="Chat with Docs", page_icon="📄")
st.title("📄 Chat with Your Documents")

# 📂 Upload documents
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
)

# Track uploaded filenames and only clear QA chain if files change
current_filenames = [f.name for f in uploaded_files] if uploaded_files else []
if 'last_uploaded_filenames' not in st.session_state:
    st.session_state.last_uploaded_filenames = []

if current_filenames != st.session_state.last_uploaded_filenames:
    st.session_state.qa = None
    st.session_state.last_uploaded_filenames = current_filenames

# Helper to load docs from in-memory files
@st.cache_data(show_spinner=False)
def load_docs_from_memory(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file.flush()
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_file.name)
            elif suffix == ".txt":
                loader = TextLoader(tmp_file.name)
            else:
                continue
            docs.extend(loader.load())
        os.unlink(tmp_file.name)
    return docs

# 🔄 Load and index documents
if st.button("📊 Index Documents"):
    st.session_state.qa = None  # Only clear QA chain when re-indexing
    if not uploaded_files:
        st.warning("Please upload at least one document to index.")
    else:
        with st.spinner("⬆️ Indexing documents..."):
            try:
                docs = load_docs_from_memory(uploaded_files)
                if not docs:
                    st.warning("No valid documents found in the uploaded files.")
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
if "qa" in st.session_state and st.session_state.qa is not None:
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
