import os

files = {
    "rag.py": '''from langchain_community.document_loaders import PyPDFLoader
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

def get_summary(model: str = "llama3.2"):
    if not os.path.exists(VECTOR_STORE_PATH):
        return "No document uploaded yet."
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke("What is this document about?")
    context = "\\n\\n".join([d.page_content for d in docs])
    llm = Ollama(model=model)
    return llm.invoke(f"Summarize this document in 3 sentences:\\n{context}")

def answer_question(question: str, chat_history: list, model: str = "llama3.2"):
    if not os.path.exists(VECTOR_STORE_PATH):
        return {"answer": "No document uploaded yet. Please upload a PDF first.", "sources": []}
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=3)
    context = "\\n\\n".join([d.page_content for d, _ in docs_and_scores])
    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"User: {msg[\'user\']}\\nAssistant: {msg[\'assistant\']}\\n"
    prompt = f"{history_text}Context:\\n{context}\\n\\nQuestion: {question}\\nAnswer:"
    llm = Ollama(model=model)
    answer = llm.invoke(prompt)
    sources = []
    for doc, score in docs_and_scores:
        meta = doc.metadata
        confidence = round((1 - float(score)) * 100, 1)
        sources.append({
            "page": meta.get("page", 0) + 1 if isinstance(meta.get("page"), int) else "?",
            "source": os.path.basename(meta.get("source", "unknown")),
            "snippet": doc.page_content[:200],
            "confidence": max(0, confidence)
        })
    return {"answer": answer, "sources": sources}

def stream_answer(question: str, chat_history: list, model: str = "llama3.2"):
    if not os.path.exists(VECTOR_STORE_PATH):
        yield "No document uploaded yet. Please upload a PDF first."
        return
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=3)
    context = "\\n\\n".join([d.page_content for d, _ in docs_and_scores])
    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"User: {msg[\'user\']}\\nAssistant: {msg[\'assistant\']}\\n"
    prompt = f"{history_text}Context:\\n{context}\\n\\nQuestion: {question}\\nAnswer:"
    llm = Ollama(model=model)
    for chunk in llm.stream(prompt):
        yield chunk
''',

    "main.py": '''from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import shutil, os, json
from rag import process_pdfs, answer_question, stream_answer, get_summary

app = FastAPI(
    title="RAG Document QA System",
    description="Upload PDFs and ask questions using a fully local RAG pipeline. No API key required.",
    version="2.0.0"
)

templates = Jinja2Templates(directory="templates")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionPayload(BaseModel):
    question: str
    chat_history: List[dict] = []
    model: str = "llama3.2"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", summary="Upload PDFs", description="Upload one or more PDFs. They will be chunked, embedded and stored in FAISS.")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    paths = []
    names = []
    for file in files:
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        paths.append(path)
        names.append(file.filename)
    chunks = process_pdfs(paths)
    return {"message": f"Processed {len(paths)} file(s) into {chunks} chunks.", "files": names}

@app.get("/uploads", summary="List uploaded files")
async def list_uploads():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    return {"files": files}

@app.get("/summary", summary="Get document summary")
async def summarize(model: str = "llama3.2"):
    summary = get_summary(model)
    return {"summary": summary}

@app.post("/ask", summary="Ask a question")
async def ask_question(payload: QuestionPayload):
    if not payload.question:
        return {"answer": "Please provide a question.", "sources": []}
    result = answer_question(payload.question, payload.chat_history, payload.model)
    return result

@app.post("/stream", summary="Stream an answer token by token")
async def stream_question(payload: QuestionPayload):
    def generate():
        for chunk in stream_answer(payload.question, payload.chat_history, payload.model):
            yield f"data: {json.dumps({'token': chunk})}\\n\\n"
        yield "data: [DONE]\\n\\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
''',

    "templates/index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Document QA</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:sans-serif;background:#f5f5f5;display:flex;justify-content:center;padding:32px 16px}
.app{width:100%;max-width:760px;display:flex;flex-direction:column;gap:16px}
h1{font-size:22px;font-weight:600}
p.sub{color:#666;font-size:14px;margin-top:4px}
.card{background:#fff;border-radius:12px;padding:20px;border:1px solid #eee}
.card h2{font-size:15px;font-weight:600;margin-bottom:12px}
.upload-zone{border:1.5px dashed #ddd;border-radius:8px;padding:16px;text-align:center;cursor:pointer;font-size:14px;color:#888;position:relative}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-zone.has-files{border-color:#4ade80;color:#166534;background:#f0fdf4}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
.btn{background:#1a1a1a;color:#fff;border:none;padding:10px 18px;border-radius:8px;font-size:14px;cursor:pointer;display:flex;align-items:center;gap:8px}
.btn:hover{background:#333}
.btn:disabled{background:#aaa;cursor:not-allowed}
.btn-sm{background:#fff;color:#1a1a1a;border:1px solid #ddd;padding:7px 12px;border-radius:8px;font-size:13px;cursor:pointer}
.btn-sm:hover{background:#f5f5f5}
.status{font-size:13px;color:#666;min-height:18px}
.summary-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px;font-size:13px;color:#166534;line-height:1.6;margin-top:10px;display:none}
.file-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.file-chip{background:#f3f4f6;border:1px solid #e5e7eb;border-radius:6px;padding:4px 10px;font-size:12px;color:#374151}
.model-row{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.model-row label{font-size:13px;color:#666}
select{padding:6px 10px;border:1px solid #ddd;border-radius:8px;font-size:13px;background:#fff}
.chat-box{display:flex;flex-direction:column;gap:12px;max-height:460px;overflow-y:auto;padding-right:4px;margin-bottom:12px}
.msg{display:flex;flex-direction:column;gap:6px}
.msg.user .bubble{background:#1a1a1a;color:#fff;align-self:flex-end;border-radius:12px 12px 2px 12px}
.msg.ai .bubble{background:#f3f4f6;color:#111;align-self:flex-start;border-radius:12px 12px 12px 2px}
.bubble{padding:10px 14px;font-size:14px;line-height:1.6;max-width:85%;white-space:pre-wrap}
.sources{display:flex;flex-direction:column;gap:6px}
.source-chip{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:8px 10px;font-size:12px;color:#1e40af;line-height:1.5}
.source-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.source-label{font-weight:600;font-size:12px}
.conf-bar-bg{height:4px;background:#dbeafe;border-radius:2px;margin-top:4px}
.conf-bar{height:4px;background:#3b82f6;border-radius:2px}
.input-row{display:flex;gap:8px}
.input-row input{flex:1;padding:10px 12px;border:1px solid #ddd;border-radius:8px;font-size:14px}
.spinner{width:14px;height:14px;border:2px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin 0.6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.empty-state{text-align:center;color:#aaa;font-size:14px;padding:32px 0}
.footer{text-align:center;font-size:12px;color:#bbb}
</style>
</head>
<body>
<div class="app">
  <div><h1>RAG Document QA System</h1><p class="sub">Upload PDFs and ask questions — runs fully locally, no API key needed.</p></div>

  <div class="card">
    <h2>1. Upload PDFs</h2>
    <div class="upload-zone" id="dropzone">
      <input type="file" id="pdfFiles" accept=".pdf" multiple onchange="handleFiles(this)">
      <span id="dropLabel">Click to choose PDFs (multiple allowed)</span>
    </div>
    <div class="row">
      <button class="btn" id="uploadBtn" onclick="uploadPDFs()" disabled>Upload & Process</button>
      <button class="btn-sm" onclick="loadUploadHistory()">View uploaded files</button>
      <span class="status" id="uploadStatus"></span>
    </div>
    <div class="file-list" id="fileList"></div>
    <div class="summary-box" id="summaryBox"></div>
  </div>

  <div class="card">
    <h2>2. Ask questions</h2>
    <div class="model-row">
      <label>Model:</label>
      <select id="modelSelect">
        <option value="llama3.2">llama3.2</option>
        <option value="mistral">mistral</option>
        <option value="llama2">llama2</option>
        <option value="phi3">phi3</option>
      </select>
      <button class="btn-sm" onclick="exportChat()">Export chat</button>
      <button class="btn-sm" onclick="clearChat()">Clear chat</button>
    </div>
    <div class="chat-box" id="chatBox">
      <div class="empty-state" id="emptyState">Upload a PDF above to get started.</div>
    </div>
    <div class="input-row">
      <input type="text" id="question" placeholder="Ask something about your documents..." onkeydown="if(event.key===\'Enter\')askQuestion()">
      <button class="btn" id="askBtn" onclick="askQuestion()">Ask</button>
    </div>
  </div>

  <p class="footer">View API docs at <a href="/docs" style="color:#888">/docs</a></p>
</div>

<script>
let chatHistory = [];
let ready = false;

function handleFiles(input) {
  const count = input.files.length;
  document.getElementById("dropzone").classList.toggle("has-files", count > 0);
  document.getElementById("dropLabel").textContent = count > 0 ? count + " file" + (count > 1 ? "s" : "") + " selected" : "Click to choose PDFs";
  document.getElementById("uploadBtn").disabled = count === 0;
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
  fetchSummary();
  loadUploadHistory();
}

async function fetchSummary() {
  const model = document.getElementById("modelSelect").value;
  const box = document.getElementById("summaryBox");
  box.style.display = "block";
  box.textContent = "Generating summary...";
  const res = await fetch("/summary?model=" + model);
  const data = await res.json();
  box.textContent = "Summary: " + data.summary;
}

async function loadUploadHistory() {
  const res = await fetch("/uploads");
  const data = await res.json();
  const list = document.getElementById("fileList");
  list.innerHTML = "";
  data.files.forEach(f => {
    const chip = document.createElement("div");
    chip.className = "file-chip";
    chip.textContent = f;
    list.appendChild(chip);
  });
  if (data.files.length > 0) ready = true;
}

async function askQuestion() {
  const input = document.getElementById("question");
  const question = input.value.trim();
  if (!question) return;
  if (!ready) { alert("Please upload a PDF first."); return; }
  const model = document.getElementById("modelSelect").value;
  input.value = "";
  appendMessage("user", question, []);

  const askBtn = document.getElementById("askBtn");
  askBtn.innerHTML = "<div class=\'spinner\'></div>";
  askBtn.disabled = true;

  const bubble = appendStreamBubble();

  const res = await fetch("/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, chat_history: chatHistory, model })
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let fullAnswer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    const lines = text.split("\\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const payload = line.slice(6);
        if (payload === "[DONE]") break;
        try {
          const parsed = JSON.parse(payload);
          fullAnswer += parsed.token;
          bubble.textContent = fullAnswer;
          document.getElementById("chatBox").scrollTop = 999999;
        } catch {}
      }
    }
  }

  const srcRes = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, chat_history: chatHistory, model })
  });
  const srcData = await srcRes.json();
  appendSources(srcData.sources || []);
  chatHistory.push({ user: question, assistant: fullAnswer });

  askBtn.innerHTML = "Ask";
  askBtn.disabled = false;
}

function appendStreamBubble() {
  const box = document.getElementById("chatBox");
  const msg = document.createElement("div");
  msg.className = "msg ai";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = "";
  msg.appendChild(bubble);
  box.appendChild(msg);
  box.scrollTop = 999999;
  return bubble;
}

function appendMessage(role, text, sources) {
  const box = document.getElementById("chatBox");
  const msg = document.createElement("div");
  msg.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);
  box.appendChild(msg);
  box.scrollTop = 999999;
}

function appendSources(sources) {
  if (!sources.length) return;
  const box = document.getElementById("chatBox");
  const lastMsg = box.lastElementChild;
  const srcDiv = document.createElement("div");
  srcDiv.className = "sources";
  sources.forEach(s => {
    const chip = document.createElement("div");
    chip.className = "source-chip";
    const conf = Math.min(100, Math.max(0, s.confidence));
    chip.innerHTML = `<div class="source-top"><span class="source-label">${s.source} — page ${s.page}</span><span style="font-size:11px">${conf}% match</span></div>${s.snippet}...<div class="conf-bar-bg"><div class="conf-bar" style="width:${conf}%"></div></div>`;
    srcDiv.appendChild(chip);
  });
  lastMsg.appendChild(srcDiv);
}

function clearChat() {
  chatHistory = [];
  document.getElementById("chatBox").innerHTML = "";
}

function exportChat() {
  if (!chatHistory.length) { alert("No chat to export yet."); return; }
  let text = "RAG Document QA - Chat Export\\n" + "=".repeat(40) + "\\n\\n";
  chatHistory.forEach((msg, i) => {
    text += `Q${i+1}: ${msg.user}\\nA${i+1}: ${msg.assistant}\\n\\n`;
  });
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "chat-export.txt";
  a.click();
}

loadUploadHistory();
</script>
</body>
</html>
''',

    "tests/test_main.py": '''import pytest
from fastapi.testclient import TestClient
from main import app
import os, io

client = TestClient(app)

def test_home():
    res = client.get("/")
    assert res.status_code == 200

def test_list_uploads():
    res = client.get("/uploads")
    assert res.status_code == 200
    assert "files" in res.json()

def test_ask_without_upload():
    res = client.post("/ask", json={"question": "What is this about?"})
    assert res.status_code == 200
    assert "answer" in res.json()

def test_ask_empty_question():
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 200
    assert res.json()["answer"] == "Please provide a question."
''',

    ".github/workflows/ci.yml": '''name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: |
          pip install fastapi uvicorn python-multipart langchain langchain-community
          pip install sentence-transformers faiss-cpu pypdf ollama jinja2 pydantic
          pip install pytest httpx
      - name: Run tests
        run: pytest tests/ -v
'''
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated: {path}")

print("\nAll improvements applied!")