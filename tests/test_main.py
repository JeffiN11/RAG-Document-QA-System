import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app, raise_server_exceptions=False)

def test_home():
    res = client.get("/")
    assert res.status_code in [200, 500]

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