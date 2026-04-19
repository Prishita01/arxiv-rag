from sentence_transformers import CrossEncoder
from config.settings import settings

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(settings.RERANKER_MODEL) 
        
    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        # Step 1: build pairs — list of [query, chunk_text] for each chunk
        if not chunks:
            return [] 
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        # Step 2: get scores from model.predict(pairs)
        scores = self.model.predict(pairs)
        
        # Step 3: attach score to each chunk
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
        
        # Step 4: sort chunks by rerank_score, highest first
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        
        # Step 5: return top_k chunks, update rank field
        for i, chunk in enumerate(chunks[:top_k], start=1):
            chunk["rank"] = i
        return chunks[:top_k]
    
if __name__ == "__main__":
    reranker = Reranker()
    query = "What are the latest advancements in machine learning?"
    chunks = [
        {"text": "A recent paper on deep learning introduces a new architecture..."},
        {"text": "An older paper discusses support vector machines..."},
        {"text": "A survey paper reviews various machine learning techniques..."},
    ]
    top_chunks = reranker.rerank(query, chunks, top_k=2)
    for chunk in top_chunks:
        print(f"Rank: {chunk['rank']}, Score: {chunk['rerank_score']:.4f}, Text: {chunk['text']}")