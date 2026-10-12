RAG Research Assistant
A Retrieval-Augmented Generation system that answers questions over research papers using a local LLM.

How it works
PDF → chunks → embeddings → FAISS index → retrieval → LLM answer
- Embeddings: `all-mpnet-base-v2` (Hugging Face)
- Vector store: FAISS
- LLM:`Qwen2.5-0.5B-Instruct` (Hugging Face)
- Framework: LangChain
- Interface:Streamlit
- 
 Run it
pip install -r requirements.txt
streamlit run app.py

Upload a research paper (PDF) and ask questions about it.
