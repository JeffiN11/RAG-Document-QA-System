# RAG Document QA System

A local AI-powered document question-answering system. Upload any PDF and ask questions about it — no API keys required, runs entirely on your machine.

## Demo
Upload a PDF → Ask a question → Get an AI-generated answer based on the document content.

## Tech Stack
- **FastAPI** — REST API backend
- **LangChain** — RAG pipeline orchestration
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2` for document embeddings
- **FAISS** — Local vector store for similarity search
- **Ollama + Llama 3.2** — Free local LLM, no API key needed
- **PyPDF** — PDF text extraction

## How It Works
1. PDF is uploaded and split into chunks
2. Each chunk is embedded using HuggingFace sentence transformers
3. Embeddings are stored in a local FAISS vector store
4. On question input, relevant chunks are retrieved and passed to Llama 3.2
5. The LLM generates an answer grounded in the document

## Setup & Run

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed

### Installation
```bash
git clone https://github.com/JeffiN11/RAG-Document-QA-System.git
cd RAG-Document-QA-System
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2
```

### Run
```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| POST | `/upload` | Upload and process a PDF |
| POST | `/ask` | Ask a question about the uploaded PDF |

## Project Structure
```
RAG-Document-QA-System/
├── main.py          # FastAPI app and routes
├── rag.py           # RAG pipeline logic
├── templates/
│   └── index.html   # Frontend UI
├── requirements.txt
└── .env.example
```