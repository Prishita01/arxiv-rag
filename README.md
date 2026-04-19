# Scholar — ArXiv Research Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system for searching and querying ArXiv research papers using hybrid retrieval, cross-encoder reranking, and a local LLM.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What It Does

Scholar lets you ask natural language questions about research papers and receive cited, grounded answers — without hallucination. Think Perplexity AI, but for ArXiv papers running entirely on your local machine for free.

**Example:** Ask *"How does sparse attention reduce the quadratic cost of transformers?"* and Scholar retrieves the most relevant chunks from papers in its database, reranks them for precision, and generates a grounded answer with clickable source citations.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              Hybrid Retrieval               │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Dense Search   │  │  Sparse Search   │  │
│  │  (ChromaDB +    │  │  (BM25 keyword   │  │
│  │  sentence-bert) │  │   matching)      │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           └──────────┬─────────┘            │
│                      ▼                      │
│            RRF Score Fusion                 │
│         (Reciprocal Rank Fusion)            │
└──────────────────────┬──────────────────────┘
                       │ Top 20 candidates
                       ▼
┌──────────────────────────────────────────────┐
│         Cross-Encoder Reranker               │
│   (ms-marco-MiniLM — reads query+chunk       │
│    together for precise relevance scoring)   │
└──────────────────────┬───────────────────────┘
                       │ Top 5 results
                       ▼
┌──────────────────────────────────────────────┐
│         Generation (Llama 3.2 via Ollama)    │
│   Grounded answer with inline [1][2] cites   │
└──────────────────────────────────────────────┘
```

### Why Hybrid Retrieval?

| Method | Strength | Weakness |
|--------|----------|----------|
| Dense (ChromaDB) | Finds semantically similar text | Misses exact keyword matches |
| Sparse (BM25) | Finds exact keyword matches | Blind to synonyms and meaning |
| **Hybrid (RRF)** | **Best of both** | **None** |

A query for *"BERT"* might return semantically related transformer papers from ChromaDB, but miss papers that literally contain the acronym. BM25 catches that. RRF fuses both ranked lists by position (not raw score), rewarding chunks that appear high in **both** lists.

### Why Cross-Encoder Reranking?

Bi-encoders (used in dense retrieval) embed query and passage **independently** — they never directly compare them. A cross-encoder reads both together, allowing full attention between every token. This catches subtle relevance that embedding similarity misses.

The tradeoff: cross-encoders are ~100x slower than bi-encoders. The solution is the two-stage pipeline: retrieve 20 candidates fast with hybrid retrieval, rerank only those 20 with the cross-encoder, return top 5. Speed + accuracy.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Data ingestion | ArXiv Python API | Free, structured metadata + abstracts |
| Storage | PostgreSQL 16 | Reliable relational store for raw papers |
| Chunking | Custom sliding window | 512-char chunks, 50-char overlap |
| Embeddings | `all-MiniLM-L6-v2` | Free, runs on CPU/M3, 384-dim vectors |
| Vector store | ChromaDB | Local persistent vector DB |
| Keyword search | BM25 (rank-bm25) | Okapi BM25 for sparse retrieval |
| Fusion | Reciprocal Rank Fusion | Position-based, score-agnostic fusion |
| Reranking | `ms-marco-MiniLM-L-6-v2` | Cross-encoder trained on MS MARCO QA |
| Generation | Llama 3.2 3B via Ollama | Free, runs locally on M3 Metal GPU |
| API | FastAPI + Uvicorn | Async Python, auto-docs at /docs |
| UI | Vanilla HTML/CSS/JS | Zero-dependency, PDF attachment support |
| Deployment | Docker + docker-compose | One-command startup, data persistence |

---

## Quickstart

### Prerequisites

- Python 3.12
- Docker Desktop
- [Ollama](https://ollama.com) installed

### 1. Clone and configure

```bash
git clone https://github.com/Prishita01/arxiv-rag.git
cd arxiv-rag
cp .env.example .env
# Edit .env with your PostgreSQL username
```

### 2. Install Ollama and pull Llama 3.2

```bash
ollama pull llama3.2
```

### 3. Start everything

```bash
# Terminal 1 — LLM inference
OLLAMA_HOST=0.0.0.0 ollama serve

# Terminal 2 — Database + API
docker-compose -f docker/docker-compose.yml up
```

### 4. Ingest papers and start searching

Open `http://localhost:8000` in your browser.

Click **Ingest papers** → enter a topic → Scholar fetches, chunks, embeds, and indexes papers from ArXiv.

Or use the API directly:

```bash
# Ingest papers
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "max_results": 20}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how does sparse attention reduce quadratic cost?"}'
```

### 5. API docs

Interactive documentation at `http://localhost:8000/docs`

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health + model name |
| `/papers` | GET | List all indexed papers |
| `/query` | POST | Run full RAG pipeline, returns answer + sources |
| `/ingest` | POST | Fetch papers from ArXiv and index them |

### Query request

```json
{
  "query": "how does attention work in transformers?",
  "n_results": 5
}
```

### Query response

```json
{
  "query": "how does attention work in transformers?",
  "answer": "Attention mechanisms allow models to focus on relevant parts...",
  "sources": [
    {
      "source_number": 1,
      "title": "Attention Is All You Need",
      "arxiv_id": "1706.03762v5",
      "text_preview": "We propose a new simple network architecture..."
    }
  ]
}
```

---

## Key Design Decisions

**Why chunk at 512 characters with 50-character overlap?**
Most embedding models (including `all-MiniLM-L6-v2`) have a 512-token context limit. Chunking ensures no content is silently truncated. The 50-character overlap prevents key sentences from being split across chunk boundaries and losing context.

**Why RRF instead of weighted score averaging?**
Dense and sparse retrievers produce scores on completely different scales — cosine similarity (0–1) vs BM25 term frequency scores (0–∞). Direct averaging is meaningless. RRF uses only rank position, which is scale-agnostic and has been shown empirically to outperform weighted averaging in most retrieval benchmarks.

**Why retrieve 20 candidates for the reranker when returning only 5?**
The cross-encoder needs a candidate pool large enough to be useful. If dense+sparse retrieval makes a mistake in the top 5, a reranker with only 5 inputs can't fix it. Retrieving 20 gives the reranker room to surface the genuinely best results, catching ranking errors from the first stage.

**Why Llama 3.2 3B instead of a larger model?**
3B parameters fit entirely in Apple M3 Pro's unified memory (18GB), loading all 29 layers onto Metal GPU. This gives ~5-10 second response times locally for free. In a RAG system, the LLM's job is summarization of retrieved text — not general knowledge — so a smaller model is sufficient when the context is high quality.

---

## Local Development (without Docker)

```bash
# Install dependencies
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL (Homebrew)
brew services start postgresql@16

# Run the API
uvicorn api.main:app --reload --port 8000
```

---

## Known Limitations

- **No conversation memory:** Each query is independent. The LLM does not have access to previous messages in a session. Conversation history is stored in the browser UI only.
- **Abstract-only retrieval:** Papers are indexed from their abstracts, not full text. Full PDF ingestion can be done via the "Attach PDF" feature in the UI.
- **BM25 rebuilds on restart:** The BM25 index is rebuilt from the database on every startup. For large corpora this adds startup time.

---

## Future Work

- [ ] Conversation memory (pass history to LLM)
- [ ] RAGAS evaluation metrics (faithfulness, answer relevancy)
- [ ] Full paper PDF ingestion pipeline
- [ ] AWS/Railway deployment
- [ ] Query expansion for improved recall

---

## Author

**Prishita** — MS Computer Science (AI/ML), Case Western Reserve University  
[GitHub](https://github.com/Prishita01) · [LinkedIn](https://linkedin.com/in/prishitagk)
