import arxiv
import requests
import time
from typing import List, Dict, Optional
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivRetriever:
    """Retrieves papers from arXiv API"""
    
    def __init__(self, max_results: int = 50):
        self.max_results = max_results
        self.client = arxiv.Client()
    
    def search(self, query: str, categories: Optional[List[str]] = None) -> List[Dict]:
        """Search arXiv for papers matching query"""
        try:
            # Build search query with categories if provided
            if categories:
                category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
                search_query = f"({query}) AND ({category_filter})"
            else:
                search_query = query
            
            search = arxiv.Search(
                query=search_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in self.client.results(search):
                results.append({
                    'id': paper.entry_id,
                    'arxiv_id': paper.get_short_id(),
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'published': paper.published.strftime("%Y-%m-%d"),
                    'updated': paper.updated.strftime("%Y-%m-%d"),
                    'categories': paper.categories,
                    'pdf_url': paper.pdf_url,
                    'source': 'arxiv'
                })
            
            logger.info(f"Retrieved {len(results)} papers from arXiv")
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving from arXiv: {e}")
            return []


class SemanticScholarRetriever:
    """Retrieves papers from Semantic Scholar API"""
    
    def __init__(self):
        self.base_url = Config.S2_BASE_URL
        self.api_key = Config.SEMANTIC_SCHOLAR_API_KEY
        self.headers = {}
        if self.api_key:
            self.headers['x-api-key'] = self.api_key
    
    def search(self, query: str, limit: int = 50) -> List[Dict]:
        """Search Semantic Scholar for papers"""
        try:
            url = f"{self.base_url}/paper/search"
            params = {
                'query': query,
                'limit': limit,
                'fields': 'paperId,title,abstract,authors,year,citationCount,influentialCitationCount,venue,url'
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for paper in data.get('data', []):
                if paper.get('abstract'):  # Only include papers with abstracts
                    results.append({
                        'id': paper.get('paperId'),
                        's2_id': paper.get('paperId'),
                        'title': paper.get('title'),
                        'authors': [author.get('name') for author in paper.get('authors', [])],
                        'abstract': paper.get('abstract'),
                        'year': paper.get('year'),
                        'citation_count': paper.get('citationCount', 0),
                        'influential_citations': paper.get('influentialCitationCount', 0),
                        'venue': paper.get('venue'),
                        'url': paper.get('url'),
                        'source': 'semantic_scholar'
                    })
            
            logger.info(f"Retrieved {len(results)} papers from Semantic Scholar")
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving from Semantic Scholar: {e}")
            return []
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Get detailed information about a specific paper"""
        try:
            url = f"{self.base_url}/paper/{paper_id}"
            params = {
                'fields': 'paperId,title,abstract,authors,year,citationCount,references,citations,embedding'
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error getting paper details: {e}")
            return None
    
    def get_recommendations(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """Get paper recommendations based on a seed paper"""
        try:
            url = f"{self.base_url}/recommendations"
            params = {
                'paperId': paper_id,
                'limit': limit,
                'fields': 'paperId,title,abstract,authors,year'
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get('recommendedPapers', [])
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []


class GoogleScholarRetriever:
    """Retrieves papers from Google Scholar via SerpAPI"""
    
    def __init__(self):
        self.api_key = Config.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"
    
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Google Scholar for papers"""
        if not self.api_key:
            logger.warning("SerpAPI key not configured, skipping Google Scholar")
            return []
        
        try:
            params = {
                'engine': 'google_scholar',
                'q': query,
                'api_key': self.api_key,
                'num': limit
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for paper in data.get('organic_results', []):
                results.append({
                    'id': paper.get('result_id'),
                    'title': paper.get('title'),
                    'authors': paper.get('publication_info', {}).get('authors', []),
                    'abstract': paper.get('snippet', ''),
                    'year': paper.get('publication_info', {}).get('year'),
                    'citation_count': paper.get('inline_links', {}).get('cited_by', {}).get('total', 0),
                    'url': paper.get('link'),
                    'source': 'google_scholar'
                })
            
            logger.info(f"Retrieved {len(results)} papers from Google Scholar")
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving from Google Scholar: {e}")
            return []


class UnifiedRetriever:
    """Unified interface for retrieving papers from multiple sources"""
    
    def __init__(self):
        self.arxiv = ArxivRetriever()
        self.semantic_scholar = SemanticScholarRetriever()
        self.google_scholar = GoogleScholarRetriever()
    
    def search_all(self, query: str, sources: List[str] = None) -> List[Dict]:
        """Search across all configured sources"""
        if sources is None:
            sources = ['arxiv', 'semantic_scholar']
        
        all_results = []
        
        if 'arxiv' in sources:
            all_results.extend(self.arxiv.search(query))
        
        if 'semantic_scholar' in sources:
            all_results.extend(self.semantic_scholar.search(query))
        
        if 'google_scholar' in sources and Config.SERPAPI_KEY:
            all_results.extend(self.google_scholar.search(query))
        
        # Deduplicate by title (simple approach)
        seen_titles = set()
        unique_results = []
        for paper in all_results:
            title_lower = paper.get('title', '').lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_results.append(paper)
        
        logger.info(f"Total unique papers retrieved: {len(unique_results)}")
        return unique_results
