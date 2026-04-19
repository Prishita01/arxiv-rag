import requests
from config.settings import settings


class Generator:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        print(f"Generator ready: {self.model} at {self.base_url}")

    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        context = "\n\n".join([
            f"[{i+1}] {chunk['text']}"
            for i, chunk in enumerate(chunks)
        ])
        prompt = f"""You are a helpful research assistant that explains \
academic papers clearly.

Use ONLY the context provided below to answer the question.
If the answer cannot be found in the context, say exactly:
"I don't have enough information in the provided papers to answer this."

When explaining, be clear and direct. Use simple language and 
analogies where helpful. Reference sources using [1], [2] notation.

CONTEXT:
{context}

QUESTION: {query}

Answer:"""

        return prompt

    def generate(self, query: str, chunks: list[dict]) -> dict:
        if not chunks:
            return {
                "answer": "No relevant papers found for your query.",
                "sources": []
            }

        # Step 1: Build the prompt from query + chunks
        prompt = self._build_prompt(query, chunks)

        # Step 2: Call Ollama API to generate answer
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model, 
            "prompt": prompt,   
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512
            }
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        # Step 3: Extract the answer text from response
        answer = response.json()["response"].strip()

        # Step 4: Build sources list 
        sources = []
        for i, chunk in enumerate(chunks):
            sources.append({
                "source_number": i + 1,
                "title": chunk.get("metadata", {}).get("title", "Unknown"),
                "arxiv_id": chunk.get("metadata", {}).get("arxiv_id", ""),
                "text_preview": chunk["text"][:200] + "..."
            })

        return {
            "answer": answer,
            "sources": sources
        }

if __name__ == "__main__":
    generator = Generator()

    # Test 1
    print("TEST 1: Dummy chunks")

    test_chunks = [
        {
            "text": "Attention mechanisms allow models to focus on relevant parts of the input. The transformer uses self-attention to compute relationships between all tokens simultaneously.",
            "metadata": {
                "title": "Attention Is All You Need",
                "arxiv_id": "1706.03762"
            }
        },
        {
            "text": "Multi-head attention runs the attention function in parallel across multiple heads. Each head learns different types of relationships between tokens.",
            "metadata": {
                "title": "Attention Is All You Need",
                "arxiv_id": "1706.03762"
            }
        }
    ]

    result = generator.generate(
        query="How does attention work in transformers?",
        chunks=test_chunks
    )

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used:")
    for s in result['sources']:
        print(f"  [{s['source_number']}] {s['title']} — {s['arxiv_id']}")

    # Test 2: full pipeline
    print("TEST 2: Full pipeline")

    from src.retriever.hybrid_retriever import HybridRetriever
    from src.reranking.reranker import Reranker

    query = "What methods are used for feature matching in computer vision?"

    retriever = HybridRetriever()
    reranker = Reranker()

    candidates = retriever.retrieve(query=query, n_results=20)
    reranked = reranker.rerank(query=query, chunks=candidates, top_k=5)
    result = generator.generate(query=query, chunks=reranked)

    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result['sources']:
        print(f"  [{s['source_number']}] {s['title']}")