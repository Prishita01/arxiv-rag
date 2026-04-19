from rank_bm25 import BM25Okapi
from src.database.postgres_client import PostgresClient
from src.preprocessing.chunker import TextChunker

class SparseRetriever:
    def __init__(self):
        self.chunks = []
        self.bm25 = None
        self._build_index()

    def _build_index(self):
        try:
            db = PostgresClient()
            db.connect()
            db.cursor.execute("""
                SELECT arxiv_id, title, abstract, authors, 
                    published_date, categories, pdf_url 
                FROM papers;
            """)
            rows = db.cursor.fetchall()
            columns = ["arxiv_id", "title", "abstract", "authors",
                    "published_date", "categories", "pdf_url"]
            papers = [dict(zip(columns, row)) for row in rows]
            db.close()

            if not papers:
                print("Warning: No papers in database. BM25 index is empty — ingest papers to enable keyword search.")
                self.chunks = []
                self.bm25 = None
                return

            chunker = TextChunker()
            self.chunks = chunker.chunk_papers(papers)
            tokenized_corpus = [chunk["text"].lower().split() for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"Built BM25 index with {len(self.chunks)} chunks.")

        except Exception as e:
            print(f"Warning: Could not build BM25 index ({e}). Starting with empty index.")
            self.chunks = []
            self.bm25 = None

    def rebuild_index(self):
        """Call this after new papers are ingested."""
        self._build_index()
        print(f"BM25 index rebuilt with {len(self.chunks)} chunks.")
    
    def retrieve(self, query: str, n_results: int = 10) -> list[dict]:
        
        if self.bm25 is None or not self.chunks:
            print("BM25 index is empty. Skipping sparse retrieval.")
            return []
        
        # STEP 1: Tokenize the query
        tokenized_query = query.lower().split()
        # STEP 2: Score every chunk
        scores = self.bm25.get_scores(tokenized_query)
        # STEP 3: Find top N indices─
        top_indices = sorted(
            range(len(scores)),     
            key=lambda i: scores[i], 
            reverse=True             
        )[:n_results]           
        
        # STEP 4: Build standardized output
        retrieved_chunks = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            retrieved_chunks.append({
                "chunk_id": f"{chunk['arxiv_id']}_chunk_{chunk['chunk_index']}",
                "text": chunk["text"],
                "metadata": {
                    "arxiv_id": chunk["arxiv_id"],
                    "title": chunk["title"],
                    "chunk_index": chunk["chunk_index"]
                },
                
                "score": float(scores[idx]),
                "rank": rank
            })
        return retrieved_chunks


if __name__ == "__main__":
    retriever = SparseRetriever()
    query = "transformer attention mechanism"
    results = retriever.retrieve(query=query, n_results=5)
    
    print(f"\nQuery: '{query}'")
    print(f"Top {len(results)} BM25 results:\n")
    for result in results:
        print(f"Rank {result['rank']} | Score: {result['score']:.4f}")
        print(f"Paper: {result['metadata']['title']}")
        print(f"Text: {result['text'][:150]}...")