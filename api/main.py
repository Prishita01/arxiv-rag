from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.retriever.hybrid_retriever import HybridRetriever
from src.reranking.reranker import Reranker
from src.generation.generator import Generator
from src.ingestion.arxiv_fetcher import ArxivFetcher
from src.database.postgres_client import PostgresClient
from src.preprocessing.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.vectorstore.chroma_client import ChromaClient
from config.settings import settings

# ── Request/Response models ───────────────────────────────────────
# Pydantic models define what JSON shape FastAPI accepts and returns
# FastAPI auto-validates — wrong shape = automatic 422 error

class QueryRequest(BaseModel):
    query: str
    n_results: int = 5   # default to 5 final results

class IngestRequest(BaseModel):
    query: str = settings.ARXIV_SEARCH_QUERY
    max_results: int = 20

# ── Startup/shutdown ──────────────────────────────────────────────
# This runs ONCE when server starts — loads all models into memory
# All requests then share these already-loaded objects
retriever = None
reranker = None
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, reranker, generator
    print("Loading models... (this takes ~10 seconds)")
    
    # Create table first — safe to run even if table already exists
    db = PostgresClient()
    db.connect()
    db.create_table()
    db.close()
    print("Database table ready.")
    
    retriever = HybridRetriever()
    reranker = Reranker()
    generator = Generator()
    print("All models loaded. Server ready.")
    yield
    print("Server shutting down.")

# ── Create FastAPI app ─────────────────────────────────────────────
app = FastAPI(
    title="ArXiv RAG API",
    description="Research paper Q&A using hybrid retrieval + reranking",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("api/static/index.html")

# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    # YOUR CODE: return {"status": "ok", "model": settings.OLLAMA_MODEL}
    return {"status": "ok", "model": settings.OLLAMA_MODEL}

@app.get("/papers")
def list_papers():
    db = PostgresClient()
    db.connect()
    db.cursor.execute(
        "SELECT arxiv_id, title, published_date FROM papers ORDER BY created_at DESC"
    )
    rows = db.cursor.fetchall()
    columns = ["arxiv_id", "title", "published_date"]
    papers = [dict(zip(columns, row)) for row in rows]
    db.close()
    return {"count": len(papers), "papers": papers}

@app.post("/query")
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        candidates = retriever.retrieve(query=request.query, n_results=20)
        reranked = reranker.rerank(query=request.query, chunks=candidates, top_k=request.n_results)
        result = generator.generate(query=request.query, chunks=reranked)
        return {
            "query": request.query,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        fetcher = ArxivFetcher()
        papers = fetcher.fetch_papers(
            query=request.query,
            max_results=request.max_results
        )
        db = PostgresClient()
        db.connect()
        db.create_table()
        db.insert_papers_batch(papers)
        db.cursor.execute(
            "SELECT arxiv_id, title, abstract, authors, published_date, categories, pdf_url FROM papers;"
        )
        rows = db.cursor.fetchall()
        columns = ["arxiv_id","title","abstract","authors","published_date","categories","pdf_url"]
        all_papers = [dict(zip(columns, r)) for r in rows]
        db.close()

        chunks = TextChunker().chunk_papers(all_papers)
        embedded = Embedder().embed_chunks(chunks)
        ChromaClient().add_chunks(embedded)

        # ← Rebuild BM25 index with new papers
        if retriever and retriever.sparse_retriever:
            retriever.sparse_retriever.rebuild_index()
            print("BM25 index rebuilt after ingestion.")

        return {
            "message": "Ingestion successful",
            "papers_fetched": len(papers),
            "chunks_stored": len(embedded)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))