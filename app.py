import os
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
    ChatHuggingFace,
)
from langchain_core.prompts import ChatPromptTemplate

UPLOAD_DIR = "uploaded_papers"
DB_DIR = "faiss_db"

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄")
st.title("📄 Research Paper Assistant")


# load models once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 300, "return_full_text": False},
    )
    return ChatHuggingFace(llm=llm)


embeddings = load_embeddings()
model = load_model()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Use ONLY the provided context to "
            "answer the question. If the answer is not in the context, say: "
            "'I could not find the answer in the documents.'",
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- sidebar: upload research papers ----------
with st.sidebar:
    st.header("Upload research papers")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("Process papers"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Reading and indexing papers..."):
                # save uploaded PDFs to a folder
                if os.path.exists(UPLOAD_DIR):
                    shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR)
                for f in uploaded_files:
                    with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
                        out.write(f.getbuffer())

                # load pdf
                loader = PyPDFDirectoryLoader(UPLOAD_DIR)
                documents = loader.load()

                # split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = splitter.split_documents(documents)

                # store in FAISS
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(DB_DIR)

                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
                )
                st.session_state.messages = []
            st.success(f"{len(chunks)} chunks stored")

# ---------- chat ----------
if st.session_state.retriever is None:
    st.info(
        "Upload your research papers from the sidebar and click 'Process papers' to start."
    )
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("sources"):
                for s in m["sources"]:
                    st.caption(s)

    query = st.chat_input("Ask a question about your papers")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(query)
                context = "\n\n".join(d.page_content for d in docs)
                response = model.invoke(
                    prompt.invoke({"context": context, "question": query})
                )
                sources = [
                    f"{d.metadata.get('source')} (page {d.metadata.get('page', 0) + 1})"
                    for d in docs
                ]
            st.write(response.content)
            for s in sources:
                st.caption(s)

        st.session_state.messages.append(
            {"role": "assistant", "content": response.content, "sources": sources}
        )
