import psycopg2
from config.settings import settings


class PostgresClient:
    def __init__(self):
        self.host = settings.DB_HOST
        self.port = settings.DB_PORT
        self.dbname = settings.DB_NAME
        self.user = settings.DB_USER
        self.password = settings.DB_PASSWORD
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )
        self.cursor = self.conn.cursor()
        print(f"Connected to PostgreSQL database: {self.dbname}")

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id SERIAL PRIMARY KEY,
                arxiv_id VARCHAR(50) UNIQUE NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,
                published_date VARCHAR(20),
                categories TEXT,
                pdf_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()
        print("Table 'papers' ready.")

    def insert_paper(self, paper_dict: dict):
        self.cursor.execute("""
            INSERT INTO papers 
                (arxiv_id, title, abstract, authors, published_date, categories, pdf_url)
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO NOTHING;
        """, (
            paper_dict["arxiv_id"],
            paper_dict["title"],
            paper_dict["abstract"],
            paper_dict["authors"],
            paper_dict["published_date"],
            paper_dict["categories"],
            paper_dict.get("pdf_url", "")
        ))
        self.conn.commit()

    def insert_papers_batch(self, papers: list[dict]):
        inserted = 0
        for paper in papers:
            self.insert_paper(paper)
            inserted += 1
        print(f"Inserted {inserted} papers into PostgreSQL.")

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("PostgreSQL connection closed.")


if __name__ == "__main__":
    from src.ingestion.arxiv_fetcher import ArxivFetcher

    # Step 1: Fetch papers from ArXiv
    fetcher = ArxivFetcher()
    papers = fetcher.fetch_papers(query=settings.ARXIV_SEARCH_QUERY, max_results=5)
    print(f"Fetched {len(papers)} papers from ArXiv")

    # Step 2: Store them in PostgreSQL
    db = PostgresClient()
    db.connect()
    db.create_table()
    db.insert_papers_batch(papers)
    db.close()