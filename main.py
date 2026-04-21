from fastapi import FastAPI, UploadFile, File, Request
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
