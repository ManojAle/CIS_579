import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """Lexical retrieval using BM25"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.metadata = []
    
    def index(self, documents: List[Dict]):
        """Index documents for BM25 search"""
        self.metadata = documents
        
        # Combine title and abstract for indexing
        texts = []
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('abstract', '')}"
            texts.append(text.lower().split())
        
        self.bm25 = BM25Okapi(texts)
        logger.info(f"Indexed {len(documents)} documents with BM25")
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict, float]]:
        """Search using BM25 and return top-k results with scores"""
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append((self.metadata[idx], float(scores[idx])))
        
        return results


class DenseRetriever:
    """Semantic retrieval using sentence embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            # Try loading with trust_remote_code for compatibility
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Error loading {model_name}: {e}")
            logger.info("Falling back to all-MiniLM-L6-v2...")
            try:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Successfully loaded fallback model")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError(
                    "Failed to load embedding model. Try:\n"
                    "1. pip install --upgrade torch transformers sentence-transformers\n"
                    "2. Or set EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 in .env"
                )
        
        self.embeddings = None
        self.metadata = []
    
    def index(self, documents: List[Dict]):
        """Index documents by computing embeddings"""
        self.metadata = documents
        
        # Combine title and abstract for embedding
        texts = []
        for doc in documents:
            text = f"{doc.get('title', '')}. {doc.get('abstract', '')}"
            texts.append(text)
        
        logger.info(f"Computing embeddings for {len(texts)} documents...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        logger.info(f"Embeddings shape: {self.embeddings.shape}")
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict, float]]:
        """Search using cosine similarity and return top-k results"""
        if self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.metadata[idx], float(similarities[idx])))
        
        return results


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense embeddings"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever(embedding_model)
    
    def index(self, documents: List[Dict]):
        """Index documents for both BM25 and dense retrieval"""
        logger.info("Indexing documents for hybrid retrieval...")
        self.bm25_retriever.index(documents)
        self.dense_retriever.index(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        alpha: float = 0.5
    ) -> List[Tuple[Dict, float]]:
        """
        Hybrid search combining BM25 and dense retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for BM25 (1-alpha for dense). 0.5 = equal weight
        """
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        dense_results = self.dense_retriever.search(query, top_k=top_k * 2)
        
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return {}
            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score - min_score == 0:
                return {doc['id']: 1.0 for doc, _ in results}
            
            normalized = {}
            for doc, score in results:
                normalized[doc['id']] = (score - min_score) / (max_score - min_score)
            return normalized
        
        bm25_normalized = normalize_scores(bm25_results)
        dense_normalized = normalize_scores(dense_results)
        
        # Combine scores
        all_doc_ids = set(bm25_normalized.keys()) | set(dense_normalized.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_normalized.get(doc_id, 0.0)
            dense_score = dense_normalized.get(doc_id, 0.0)
            combined_scores[doc_id] = alpha * bm25_score + (1 - alpha) * dense_score
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Retrieve full documents
        id_to_doc = {}
        for doc, _ in bm25_results + dense_results:
            id_to_doc[doc['id']] = doc
        
        results = []
        for doc_id, score in sorted_ids:
            if doc_id in id_to_doc:
                results.append((id_to_doc[doc_id], score))
        
        logger.info(f"Hybrid search returned {len(results)} results")
        return results


class Reranker:
    """Re-rank retrieved documents using cross-encoder or other methods"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker - using sentence-transformers for simplicity"""
        logger.info(f"Loading reranker model: {model_name}")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.use_cross_encoder = True
        except:
            logger.warning("CrossEncoder not available, using simple scoring")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.use_cross_encoder = False
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Dict, float]],
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """Re-rank documents and return top-k"""
        if not documents:
            return []
        
        docs = [doc for doc, _ in documents]
        
        if self.use_cross_encoder:
            # Use cross-encoder for reranking
            pairs = [[query, f"{doc.get('title', '')}. {doc.get('abstract', '')}"] for doc in docs]
            scores = self.model.predict(pairs)
        else:
            # Fallback: use bi-encoder similarity
            query_emb = self.model.encode([query])[0]
            doc_texts = [f"{doc.get('title', '')}. {doc.get('abstract', '')}" for doc in docs]
            doc_embs = self.model.encode(doc_texts)
            
            scores = np.dot(doc_embs, query_emb) / (
                np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
            )
        
        # Sort by reranking score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        reranked = []
        for idx in ranked_indices:
            reranked.append((docs[idx], float(scores[idx])))
        
        logger.info(f"Reranked to top {len(reranked)} documents")
        return reranked
