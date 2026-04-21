from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os

VECTOR_STORE_PATH = "vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdfs(file_paths: list):
    all_chunks = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return len(all_chunks)

def answer_question(question: str, chat_history: list):
    if not os.path.exists(VECTOR_STORE_PATH):
        return {"answer": "No document uploaded yet. Please upload a PDF first.", "sources": []}

    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"

    prompt = f"{history_text}Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    llm = Ollama(model="llama3.2")
    answer = llm.invoke(prompt)

    sources = []
    for doc in docs:
        meta = doc.metadata
        sources.append({
            "page": meta.get("page", 0) + 1 if isinstance(meta.get("page"), int) else "?",
            "source": os.path.basename(meta.get("source", "unknown")),
            "snippet": doc.page_content[:200]
        })

    return {"answer": answer, "sources": sources}