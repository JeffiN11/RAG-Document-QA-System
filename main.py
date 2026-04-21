from fastapi import FastAPI, UploadFile, File, Request
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
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
