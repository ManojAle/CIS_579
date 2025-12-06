import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for Research Assistant AI"""
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    
    # Models
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Retrieval Settings
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
    TOP_K_RETRIEVAL = 20
    TOP_K_RERANK = 10
    
    # arXiv Settings
    ARXIV_MAX_RESULTS = 50
    ARXIV_CATEGORIES = [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR",
        "stat.ML", "math.ST", "physics.data-an"
    ]
    
    # Semantic Scholar Settings
    S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
    S2_RATE_LIMIT = 100  # requests per 5 minutes
    
    # Vector Store Settings
    VECTOR_DB_PATH = "./data/vector_store"
    CACHE_DIR = "./data/cache"
    
    # Generation Settings
    MAX_TOKENS = 4000
    TEMPERATURE = 0.3
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required")
        return True
