import os

files = {
    "rag.py": '''from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os

VECTOR_STORE_PATH = "vectorstore"

def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return len(chunks)

def answer_question(question: str):
    if not os.path.exists(VECTOR_STORE_PATH):
        return "No document uploaded yet. Please upload a PDF first."
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = Ollama(model="llama3.2")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return qa_chain.invoke({"query": question})["result"]
''',

    "main.py": '''from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import shutil, os
from rag import process_pdf, answer_question

app = FastAPI()
templates = Jinja2Templates(directory="templates")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = process_pdf(file_path)
    return {"message": f"PDF processed successfully into {chunks} chunks."}

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    if not question:
        return {"answer": "Please provide a question."}
    answer = answer_question(question)
    return {"answer": answer}
''',

    "templates/index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RAG Document QA</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: sans-serif; background: #f5f5f5; display: flex; justify-content: center; padding: 40px 16px; }
        .container { background: white; border-radius: 12px; padding: 32px; max-width: 680px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
        h1 { font-size: 22px; margin-bottom: 8px; }
        p.sub { color: #666; font-size: 14px; margin-bottom: 24px; }
        .section { margin-bottom: 24px; }
        label { font-size: 13px; font-weight: 500; display: block; margin-bottom: 6px; }
        input[type=file] { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; }
        input[type=text] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; }
        button { background: #1a1a1a; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-size: 14px; cursor: pointer; margin-top: 8px; }
        button:hover { background: #333; }
        .answer-box { background: #f9f9f9; border: 1px solid #eee; border-radius: 8px; padding: 16px; font-size: 14px; line-height: 1.6; min-height: 60px; margin-top: 12px; color: #333; }
        .status { font-size: 13px; color: #888; margin-top: 8px; }
    </style>
</head>
<body>
<div class="container">
    <h1>RAG Document QA System</h1>
    <p class="sub">Upload a PDF and ask questions about it using AI.</p>

    <div class="section">
        <label>1. Upload a PDF</label>
        <input type="file" id="pdfFile" accept=".pdf" />
        <button onclick="uploadPDF()">Upload & Process</button>
        <div class="status" id="uploadStatus"></div>
    </div>

    <div class="section">
        <label>2. Ask a question</label>
        <input type="text" id="question" placeholder="e.g. What is this document about?" />
        <button onclick="askQuestion()">Ask</button>
        <div class="answer-box" id="answerBox">Your answer will appear here...</div>
    </div>
</div>

<script>
async function uploadPDF() {
    const file = document.getElementById("pdfFile").files[0];
    if (!file) return alert("Please select a PDF file.");
    const status = document.getElementById("uploadStatus");
    status.textContent = "Uploading and processing...";
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    status.textContent = data.message;
}

async function askQuestion() {
    const question = document.getElementById("question").value;
    if (!question) return alert("Please enter a question.");
    const box = document.getElementById("answerBox");
    box.textContent = "Thinking...";
    const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
    });
    const data = await res.json();
    box.textContent = data.answer;
}
</script>
</body>
</html>
''',

    ".env.example": '''# No API keys needed - this project runs fully locally using Ollama
''',

    ".gitignore": '''venv/
__pycache__/
uploads/
vectorstore/
*.pyc
.env
''',

    "requirements.txt": '''fastapi
uvicorn
python-multipart
langchain
langchain-community
sentence-transformers
faiss-cpu
pypdf
ollama
jinja2
'''
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {path}")

print("\nAll files created successfully!")