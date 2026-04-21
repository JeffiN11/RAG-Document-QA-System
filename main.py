from fastapi import FastAPI, UploadFile, File
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
