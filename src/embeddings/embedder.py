from sentence_transformers import SentenceTransformer
from config.settings import settings

class Embedder:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        print(f"Loaded embedding model: {self.model_name}")

    def embed_text(self, text: list[str]) -> list[list[float]]:
        vectors = self.model.encode(text, show_progress_bar=True)
        return vectors.tolist()
    
    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        # Extract the text to embed from each chunk dict
        texts = [chunk["text"] for chunk in chunks]
        vectors = self.model.encode(texts)
        
        # Add the embedding back into each chunk dict
        for i, chunk_dict in enumerate(chunks):
            chunk_dict["embedding"] = vectors[i].tolist()
        return chunks

if __name__ == "__main__":
    from src.preprocessing.chunker import TextChunker
    from src.database.postgres_client import PostgresClient

    # Get chunks from PostgreSQL
    db = PostgresClient()
    db.connect()
    db.cursor.execute("SELECT arxiv_id, title, abstract, authors, published_date, categories, pdf_url FROM papers LIMIT 3;")
    rows = db.cursor.fetchall()
    columns = ["arxiv_id", "title", "abstract", "authors", "published_date", "categories", "pdf_url"]
    papers = [dict(zip(columns, row)) for row in rows]
    db.close()

    # Chunk them
    chunker = TextChunker()
    chunks = chunker.chunk_papers(papers)
    print(f"Chunks to embed: {len(chunks)}")

    # Embed them
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    print(f"\nFirst chunk embedding:")
    print(f"Text preview: {embedded_chunks[0]['text'][:100]}...")
    print(f"Embedding dimensions: {len(embedded_chunks[0]['embedding'])}")
    print(f"First 5 values: {embedded_chunks[0]['embedding'][:5]}")