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
    context = "\\n\\n".join([d.page_content for d in docs])
    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"User: {msg[\'user\']}\\nAssistant: {msg[\'assistant\']}\\n"
    prompt = f"{history_text}Context:\\n{context}\\n\\nQuestion: {question}\\nAnswer:"
    llm = Ollama(model="llama3.2")
    answer = llm.invoke(prompt)
    sources = []
    for doc in docs:
        meta = doc.metadata
        sources.append({
            "page": meta.get("page", "?") + 1 if isinstance(meta.get("page"), int) else "?",
            "source": os.path.basename(meta.get("source", "unknown")),
            "snippet": doc.page_content[:200]
        })
    return {"answer": answer, "sources": sources}
''',

    "main.py": '''from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import shutil, os
from rag import process_pdfs, answer_question

app = FastAPI(
    title="RAG Document QA System",
    description="Upload PDFs and ask questions using a fully local RAG pipeline. No API key required.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionPayload(BaseModel):
    question: str
    chat_history: List[dict] = []

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", summary="Upload PDFs", description="Upload one or more PDF files. They will be chunked, embedded, and stored in a local FAISS vector store.")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    paths = []
    for file in files:
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        paths.append(path)
    chunks = process_pdfs(paths)
    return {"message": f"Processed {len(paths)} file(s) into {chunks} chunks."}

@app.post("/ask", summary="Ask a question", description="Ask a question about the uploaded documents. Optionally pass chat_history for multi-turn conversation.")
async def ask_question(payload: QuestionPayload):
    if not payload.question:
        return {"answer": "Please provide a question.", "sources": []}
    result = answer_question(payload.question, payload.chat_history)
    return result
''',

    "templates/index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Document QA</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:sans-serif;background:#f5f5f5;display:flex;justify-content:center;padding:32px 16px}
.app{width:100%;max-width:720px;display:flex;flex-direction:column;gap:16px}
h1{font-size:22px;font-weight:600}
p.sub{color:#666;font-size:14px;margin-top:4px}
.card{background:#fff;border-radius:12px;padding:20px;border:1px solid #eee}
.card h2{font-size:15px;font-weight:600;margin-bottom:12px}
.upload-zone{border:1.5px dashed #ddd;border-radius:8px;padding:16px;text-align:center;cursor:pointer;font-size:14px;color:#888;position:relative}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-zone.has-files{border-color:#4ade80;color:#166534;background:#f0fdf4}
.btn{background:#1a1a1a;color:#fff;border:none;padding:10px 18px;border-radius:8px;font-size:14px;cursor:pointer;display:flex;align-items:center;gap:8px}
.btn:hover{background:#333}
.btn:disabled{background:#aaa;cursor:not-allowed}
.btn-outline{background:#fff;color:#1a1a1a;border:1px solid #ddd;padding:8px 14px;border-radius:8px;font-size:13px;cursor:pointer}
.btn-outline:hover{background:#f5f5f5}
.status{font-size:13px;color:#666;margin-top:8px;min-height:18px}
.chat-box{display:flex;flex-direction:column;gap:12px;max-height:420px;overflow-y:auto;padding-right:4px}
.msg{display:flex;flex-direction:column;gap:4px}
.msg.user .bubble{background:#1a1a1a;color:#fff;align-self:flex-end;border-radius:12px 12px 2px 12px}
.msg.ai .bubble{background:#f3f4f6;color:#111;align-self:flex-start;border-radius:12px 12px 12px 2px}
.bubble{padding:10px 14px;font-size:14px;line-height:1.6;max-width:85%}
.sources{display:flex;flex-direction:column;gap:6px;margin-top:4px}
.source-chip{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:8px 10px;font-size:12px;color:#1e40af;line-height:1.5}
.source-label{font-weight:600;margin-bottom:2px}
.input-row{display:flex;gap:8px;margin-top:8px}
.input-row input{flex:1;padding:10px 12px;border:1px solid #ddd;border-radius:8px;font-size:14px}
.spinner{width:14px;height:14px;border:2px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin 0.6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.empty-state{text-align:center;color:#aaa;font-size:14px;padding:32px 0}
</style>
</head>
<body>
<div class="app">
  <div>
    <h1>RAG Document QA System</h1>
    <p class="sub">Upload PDFs and ask questions — runs fully locally, no API key needed.</p>
  </div>

  <div class="card">
    <h2>1. Upload PDFs</h2>
    <div class="upload-zone" id="dropzone">
      <input type="file" id="pdfFiles" accept=".pdf" multiple onchange="handleFiles(this)">
      <span id="dropLabel">Click to choose PDFs (multiple allowed)</span>
    </div>
    <div style="display:flex;gap:8px;margin-top:10px;align-items:center">
      <button class="btn" id="uploadBtn" onclick="uploadPDFs()" disabled>Upload & Process</button>
      <span class="status" id="uploadStatus"></span>
    </div>
  </div>

  <div class="card">
    <h2>2. Ask questions</h2>
    <div class="chat-box" id="chatBox">
      <div class="empty-state" id="emptyState">Upload a PDF above to get started.</div>
    </div>
    <div class="input-row">
      <input type="text" id="question" placeholder="Ask something about your documents..." onkeydown="if(event.key==='Enter')askQuestion()">
      <button class="btn" id="askBtn" onclick="askQuestion()">Ask</button>
    </div>
    <div style="display:flex;justify-content:flex-end;margin-top:8px">
      <button class="btn-outline" onclick="clearChat()">Clear chat</button>
    </div>
  </div>

  <p style="text-align:center;font-size:12px;color:#bbb">View API docs at <a href="/docs" style="color:#888">/docs</a></p>
</div>

<script>
let chatHistory = [];
let ready = false;

function handleFiles(input) {
  const count = input.files.length;
  const zone = document.getElementById("dropzone");
  const label = document.getElementById("dropLabel");
  const btn = document.getElementById("uploadBtn");
  if (count > 0) {
    zone.classList.add("has-files");
    label.textContent = count + " file" + (count > 1 ? "s" : "") + " selected";
    btn.disabled = false;
  }
}

async function uploadPDFs() {
  const files = document.getElementById("pdfFiles").files;
  if (!files.length) return;
  const btn = document.getElementById("uploadBtn");
  const status = document.getElementById("uploadStatus");
  btn.disabled = true;
  status.textContent = "Processing...";
  const formData = new FormData();
  for (const f of files) formData.append("files", f);
  const res = await fetch("/upload", { method: "POST", body: formData });
  const data = await res.json();
  status.textContent = data.message;
  ready = true;
  document.getElementById("emptyState")?.remove();
}

async function askQuestion() {
  const input = document.getElementById("question");
  const question = input.value.trim();
  if (!question) return;
  if (!ready) { alert("Please upload a PDF first."); return; }

  input.value = "";
  appendMessage("user", question, []);

  const askBtn = document.getElementById("askBtn");
  askBtn.innerHTML = "<div class=\'spinner\'></div>";
  askBtn.disabled = true;

  const thinkingId = "thinking-" + Date.now();
  appendMessage("ai", "Thinking...", [], thinkingId);

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, chat_history: chatHistory })
  });
  const data = await res.json();

  document.getElementById(thinkingId)?.remove();
  appendMessage("ai", data.answer, data.sources || []);
  chatHistory.push({ user: question, assistant: data.answer });

  askBtn.innerHTML = "Ask";
  askBtn.disabled = false;
}

function appendMessage(role, text, sources, id) {
  const box = document.getElementById("chatBox");
  const msg = document.createElement("div");
  msg.className = "msg " + role;
  if (id) msg.id = id;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);

  if (sources && sources.length > 0) {
    const srcDiv = document.createElement("div");
    srcDiv.className = "sources";
    sources.forEach(s => {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      chip.innerHTML = "<div class=\'source-label\'>" + s.source + " — page " + s.page + "</div>" + s.snippet + "...";
      srcDiv.appendChild(chip);
    });
    msg.appendChild(srcDiv);
  }

  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
}

function clearChat() {
  chatHistory = [];
  const box = document.getElementById("chatBox");
  box.innerHTML = "";
}
</script>
</body>
</html>
''',

    "Dockerfile": '''FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads vectorstore

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
''',

    "docker-compose.yml": '''version: "3.8"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./vectorstore:/app/vectorstore
    extra_hosts:
      - "host.docker.internal:host-gateway"
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
pydantic
'''
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated: {path}")

print("\nAll improvements applied!")