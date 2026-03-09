from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import create_embeddings, query_rag, decide_route, call_llm

app = FastAPI()

# =====================
# SIMPLE SESSION MEMORY
# =====================
session_memory = {}

class AskRequest(BaseModel):
    query: str
    session_id: str | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

@app.on_event("startup")
def startup_event():
    print("Creating embeddings from data folder...")
    create_embeddings()
    print("✅ Embeddings ready!")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        route = decide_route(req.query)
        sources = []

        if route == "rag":
            context, sources = query_rag(req.query)
            prompt = f"""
Use the following documents to answer the question.

Documents:
{context}

Question:
{req.query}
"""
        else:
            prompt = req.query

        # 🔑 LLM call (OpenRouter)
        answer = call_llm(prompt)

        if req.session_id:
            session_memory.setdefault(req.session_id, []).append({
                "query": req.query,
                "answer": answer
            })

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        # ✅ Proper API error (NOT 200 OK)
        raise HTTPException(
            status_code=500,
            detail=f"LLM processing failed: {str(e)}"
        )

