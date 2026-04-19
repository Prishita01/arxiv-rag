import chromadb
from config.settings import settings

class ChromaClient:
    def __init__(self, collection_name: str = "arxiv_papers"):
        self.client = settings.CHROMA_PATH
        self.collection_name = collection_name
        self.db = chromadb.PersistentClient(path=self.client)
        self.collection = self.db.get_or_create_collection(name=self.collection_name)
        
    def add_chunks(self, chunks: list[dict]):
        # Prepare data for ChromaDB
        ids = [f"{chunk['arxiv_id']}_{chunk['chunk_index']}" for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        metadatas = [
            {
                "arxiv_id": chunk["arxiv_id"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
                "authors": chunk["authors"],
                "published_date": chunk["published_date"],
                "categories": chunk["categories"],
                "pdf_url": chunk.get("pdf_url", "")
            }
            for chunk in chunks
        ]
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        
    def query(self, query_embedding: list[float], n_results: int = 5) -> dict:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results
    
if __name__ == "__main__":
    from src.preprocessing.chunker import TextChunker
    from src.embeddings.embedder import Embedder
    from src.database.postgres_client import PostgresClient

    # Step 1: Fetch papers from PostgreSQL
    db = PostgresClient()
    db.connect()
    db.cursor.execute("SELECT arxiv_id, title, abstract, authors, published_date, categories, pdf_url FROM papers LIMIT 3;")
    rows = db.cursor.fetchall()
    columns = ["arxiv_id", "title", "abstract", "authors", "published_date", "categories", "pdf_url"]
    papers = [dict(zip(columns, row)) for row in rows]
    db.close()

    # Step 2: Chunk them
    chunker = TextChunker()
    chunks = chunker.chunk_papers(papers)

    # Step 3: Embed them
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    # Step 4: Add to ChromaDB
    chroma_client = ChromaClient()
    chroma_client.add_chunks(embedded_chunks)
    
    print(f"Added {len(embedded_chunks)} chunks to ChromaDB collection '{chroma_client.collection_name}'.")