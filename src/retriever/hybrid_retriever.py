from src.retriever.dense_retriever import DenseRetriever
from src.retriever.sparse_retriever import SparseRetriever

class HybridRetriever:
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        
    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        dense_results = self.dense_retriever.retrieve(query=query, n_results=n_results)
        sparse_results = self.sparse_retriever.retrieve(query=query, n_results=n_results)
        
        all_results = {r["chunk_id"]: r for r in dense_results + sparse_results}
        
        # RRF scoring
        rrf_scores = {}
        for result in dense_results:
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1/(60 + result["rank"])
        for result in sparse_results:
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1/(60 + result["rank"])
        
        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        seen_texts = set()
        final_results = []
        rank = 1
        for chunk_id, rrf_score in sorted_chunks:
            result = all_results[chunk_id]
            text_fingerprint = result["text"][:100]
            if text_fingerprint not in seen_texts:
                seen_texts.add(text_fingerprint)
                final_results.append({
                    "rank": rank,
                    "rrf_score": rrf_score,
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"]
                })
                rank += 1
            if rank > n_results:
                break
        
        print(f"Dense found: {len(dense_results)} | Sparse found: {len(sparse_results)} | Combined: {len(final_results)}")
        return final_results
    
if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "transformer attention mechanism"
    results = retriever.retrieve(query=query, n_results=5)
    print(f"Query: '{query}'\n")
    for r in results:
        print(f"Rank {r['rank']} | RRF Score: {r['rrf_score']:.4f}")
        print(f"Paper: {r['metadata']['title']}")
        print(f"Text: {r['text'][:150]}...")