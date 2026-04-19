class TextChunker:
    def __init__(self, chunk_size : int = 512, chunk_overlap : int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        start = 0
        chunks = []
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = start + self.chunk_size - self.chunk_overlap
        return chunks
    
    def chunk_paper(self, paper_dict: dict) -> list[dict]:
        chunks = self.chunk_text(paper_dict["abstract"])
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "chunk_index": i,
                "arxiv_id": paper_dict["arxiv_id"],
                "title": paper_dict["title"],
                "text": chunk,
                "authors": paper_dict["authors"],
                "published_date": paper_dict["published_date"],
                "categories": paper_dict["categories"],
                "pdf_url": paper_dict.get("pdf_url", "")
            }
            chunk_dicts.append(chunk_dict)
        return chunk_dicts

    def chunk_papers(self, papers: list[dict]) -> list[dict]:
        all_chunks = []
        for paper in papers:
            chunks = self.chunk_paper(paper)
            all_chunks.extend(chunks)
        return all_chunks
        
if __name__ == "__main__":
    from src.database.postgres_client import PostgresClient
    
    db = PostgresClient()
    db.connect()
    
    # Fetch papers from PostgreSQL directly
    db.cursor.execute("SELECT arxiv_id, title, abstract, authors, published_date, categories, pdf_url FROM papers LIMIT 5;")
    rows = db.cursor.fetchall()
    columns = ["arxiv_id", "title", "abstract", "authors", "published_date", "categories", "pdf_url"]
    papers = [dict(zip(columns, row)) for row in rows]
    db.close()
    
    # Chunk them
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    all_chunks = chunker.chunk_papers(papers)
    
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"\nFirst chunk preview:")
    print(f"Paper: {all_chunks[0]['title']}")
    print(f"Chunk index: {all_chunks[0]['chunk_index']}")
    print(f"Text: {all_chunks[0]['text'][:200]}...")