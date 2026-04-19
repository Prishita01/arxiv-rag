from src.retriever.hybrid_retriever import HybridRetriever
from src.reranking.reranker import Reranker

def run_pipeline(query: str, retrieve_k: int = 20, final_k: int = 5):
    print(f"Query: '{query}'")
    
    # Step 1: Hybrid retrieval — cast wide net
    print(f"\n[1] Hybrid retrieval (fetching {retrieve_k} candidates)...")
    retriever = HybridRetriever()
    candidates = retriever.retrieve(query=query, n_results=retrieve_k)
    print(f"    Got {len(candidates)} candidates")
    
    # Step 2: Rerank — precision pass
    print(f"\n[2] Cross-encoder reranking (top {final_k})...")
    reranker = Reranker()
    final_results = reranker.rerank(
        query=query,
        chunks=candidates,
        top_k=final_k
    )
    
    # Step 3: Display results
    print(f"\n[3] Final results:")
    for r in final_results:
        print(f"\nRank {r['rank']} | Rerank score: {r['rerank_score']:.4f}")
        print(f"Paper: {r.get('metadata', {}).get('title', 'N/A')}")
        print(f"Text:  {r['text'][:200]}...")
    
    return final_results

if __name__ == "__main__":
    queries = [
        "transformer attention mechanism",
        "deep learning optimization methods",
        "computer vision feature matching"
    ]
    for query in queries:
        run_pipeline(query, retrieve_k=20, final_k=3)