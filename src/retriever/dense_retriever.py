from src.vectorstore.chroma_client import ChromaClient
from src.embeddings.embedder import Embedder

class DenseRetriever:
    def __init__(self, collection_name: str = "arxiv_papers"):
        self.chroma_client = ChromaClient(collection_name=collection_name)
        self.embedder = Embedder()
        
    def retrieve(self, query: str, n_results: int = 10) -> dict:
        query_embedding = self.embedder.embed_text(query)
        results = self.chroma_client.query(query_embedding=query_embedding, n_results=n_results)
        retrieved_chunks = []
        for idx, (doc_id, doc_text, doc_metadata, doc_score) in enumerate(zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])):
            retrieved_chunks.append({
                "chunk_id": doc_id,
                "text": doc_text,
                "metadata": doc_metadata,
                "score": 1-doc_score,
                "rank": idx + 1
            })
        return retrieved_chunks
    
if __name__ == "__main__":
    retriever = DenseRetriever()
    query = "What are the latest advancements in natural language processing?"
    results = retriever.retrieve(query=query, n_results=10)
    for result in results:
        print(f"Rank: {result['rank']}, Score: {result['score']:.4f}, Chunk ID: {result['chunk_id']}")
        print(f"Text: {result['text'][:200]}...")
        print(f"Metadata: {result['metadata']}")