from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pipeline.rag_pipeline import run_rag_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API Schema ===
class QueryRequest(BaseModel):
    query: str
    history: list

@app.post("/ask")
async def ask(req: QueryRequest):
    """
    Accepts a query and history, runs the RAG pipeline, and returns the answer and sources.
    """
    result = run_rag_pipeline(req.query, req.history)
    #print("Returning result:", result)
    return result


# === Health Check Endpoint ===
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})
