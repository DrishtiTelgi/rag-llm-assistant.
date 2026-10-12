from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
    ChatHuggingFace,
)
from langchain_core.prompts import ChatPromptTemplate

# load the saved index (no re-embedding)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local(
    "faiss_db", embeddings, allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
)

# free local LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 300, "return_full_text": False},
)
model = ChatHuggingFace(llm=llm)

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

print("RAG system ready. Press 0 to exit.")

while True:
    query = input("\nYou: ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    response = model.invoke(prompt.invoke({"context": context, "question": query}))
    print(f"\nAI: {response.content}")

    # show sources
    for d in docs:
        print(f"  - {d.metadata.get('source')} (page {d.metadata.get('page', 0) + 1})")
