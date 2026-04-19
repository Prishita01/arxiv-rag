import arxiv
from config.settings import settings

class ArxivFetcher:
    def __init__(self):
        self.client = arxiv.Client()
        
    def fetch_papers(self, query:str, max_results:int) -> list[dict]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
    
        papers = []

        for paper in self.client.results(search):
            paper_dict = {
                "arxiv_id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "authors": ", ".join([author.name for author in paper.authors]),
                "abstract": paper.summary,
                "published_date": paper.published.strftime("%Y-%m-%d"),
                "categories": ", ".join(paper.categories),
                "updated": paper.updated.isoformat(),
                "pdf_url": paper.pdf_url
            }
            papers.append(paper_dict)
        return papers
    
if __name__ == "__main__":
    fetcher = ArxivFetcher()
    papers = fetcher.fetch_papers(query=settings.ARXIV_SEARCH_QUERY, max_results=3)
    for p in papers:
        print(f"ID: {p['arxiv_id']}")
        print(f"Title: {p['title']}")
        print(f"Authors: {p['authors']}")
        print(f"Date: {p['published_date']}")
        print(f"Abstract preview: {p['abstract'][:150]}...")