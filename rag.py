import os
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# =====================
# CONFIG
# =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-7b-instruct"

DATA_FOLDER = "data"

# =====================
# GLOBAL OBJECTS
# =====================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = None
documents = []  # [{text, source}]

# =====================
# AGENT DECISION LOGIC
# =====================
def decide_route(query: str) -> str:
    keywords = ["policy", "leave", "vacation", "faq", "document", "technical"]
    for k in keywords:
        if k in query.lower():
            return "rag"
    return "llm"

# =====================
# EMBEDDINGS + FAISS
# =====================
def create_embeddings():
    global faiss_index, documents

    documents.clear()
    texts = []

    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({"text": content, "source": file})
                texts.append(content)

    embeddings = embedder.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)

    print(f"✅ Created {len(texts)} embeddings using FAISS")

# =====================
# RETRIEVE DOCUMENTS
# =====================
def query_rag(query: str, top_k: int = 2):
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_vec, top_k)

    results = []
    sources = []

    for idx in indices[0]:
        results.append(documents[idx]["text"])
        sources.append(documents[idx]["source"])

    return "\n\n".join(results), sources

# =====================
# OPENROUTER LLM CALL
# =====================
def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI-Agent-Assignment"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Error: {response.text}"

    return response.json()["choices"][0]["message"]["content"]
