"""
Evaluation module for Research Assistant AI
Implements IR metrics (Recall@k, Precision@k, nDCG, MRR) and RAGAS metrics
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IRMetrics:
    """Information Retrieval evaluation metrics"""
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Recall@K: Proportion of relevant documents retrieved in top-k
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(retrieved_k & relevant_set)
        return hits / len(relevant_set)
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Precision@K: Proportion of retrieved documents that are relevant in top-k
        """
        if not retrieved_ids[:k]:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(retrieved_k & relevant_set)
        return hits / k
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        MRR: Reciprocal of the rank of the first relevant document
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        
        Args:
            retrieved_ids: List of retrieved document IDs in order
            relevance_scores: Dict mapping doc_id to relevance score (0-3 scale typical)
            k: Number of top results to consider
        """
        def dcg(scores: List[float], k: int) -> float:
            """Calculate DCG@k"""
            scores_k = scores[:k]
            return sum((2**score - 1) / np.log2(i + 2) for i, score in enumerate(scores_k))
        
        # Get relevance scores for retrieved docs
        retrieved_scores = [relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_ids]
        
        # Calculate DCG for retrieved documents
        dcg_score = dcg(retrieved_scores, k)
        
        # Calculate ideal DCG (sorted by relevance)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)
        idcg_score = dcg(ideal_scores, k)
        
        # Return normalized DCG
        if idcg_score == 0:
            return 0.0
        return dcg_score / idcg_score
    
    @staticmethod
    def evaluate_retrieval(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        relevance_scores: Dict[str, float] = None,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Comprehensive retrieval evaluation
        
        Returns dict with all metrics
        """
        results = {}
        
        for k in k_values:
            results[f'recall@{k}'] = IRMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
            results[f'precision@{k}'] = IRMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
            
            if relevance_scores:
                results[f'ndcg@{k}'] = IRMetrics.ndcg_at_k(retrieved_ids, relevance_scores, k)
        
        results['mrr'] = IRMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        
        return results


class RAGASMetrics:
    """RAGAS evaluation metrics for RAG systems"""
    
    @staticmethod
    def answer_faithfulness_simple(
        answer: str,
        contexts: List[str],
        llm=None
    ) -> float:
        """
        Simplified faithfulness check: Does the answer contain claims from contexts?
        
        In production, this would use an LLM to verify each claim.
        This is a simplified version that checks for text overlap.
        """
        if not contexts or not answer:
            return 0.0
        
        # Simple heuristic: check if key phrases from answer appear in contexts
        answer_lower = answer.lower()
        context_text = " ".join(contexts).lower()
        
        # Split answer into sentences
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Count how many sentences have significant overlap with contexts
        supported = 0
        for sentence in sentences:
            words = set(sentence.split())
            # Check if at least 30% of words appear in contexts
            overlap = sum(1 for word in words if word in context_text)
            if overlap / max(len(words), 1) > 0.3:
                supported += 1
        
        return supported / len(sentences)
    
    @staticmethod
    def context_precision(
        contexts: List[str],
        relevant_contexts: List[str]
    ) -> float:
        """
        Context Precision: Proportion of retrieved contexts that are relevant
        """
        if not contexts:
            return 0.0
        
        relevant_set = set(relevant_contexts)
        hits = sum(1 for ctx in contexts if ctx in relevant_set)
        
        return hits / len(contexts)
    
    @staticmethod
    def context_recall(
        contexts: List[str],
        relevant_contexts: List[str]
    ) -> float:
        """
        Context Recall: Proportion of relevant contexts that were retrieved
        """
        if not relevant_contexts:
            return 0.0
        
        retrieved_set = set(contexts)
        hits = sum(1 for ctx in relevant_contexts if ctx in retrieved_set)
        
        return hits / len(relevant_contexts)
    
    @staticmethod
    def answer_relevancy_simple(
        question: str,
        answer: str
    ) -> float:
        """
        Simplified answer relevancy: Keyword overlap between question and answer
        
        In production, this would use embeddings or LLM scoring.
        """
        if not question or not answer:
            return 0.0
        
        # Extract keywords (simple approach)
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        q_words -= stopwords
        a_words -= stopwords
        
        if not q_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(q_words & a_words)
        return overlap / len(q_words)
    
    @staticmethod
    def evaluate_rag(
        question: str,
        answer: str,
        contexts: List[str],
        relevant_contexts: List[str] = None
    ) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation
        
        Returns dict with RAGAS-inspired metrics
        """
        results = {}
        
        # Faithfulness
        results['faithfulness'] = RAGASMetrics.answer_faithfulness_simple(answer, contexts)
        
        # Relevancy
        results['answer_relevancy'] = RAGASMetrics.answer_relevancy_simple(question, answer)
        
        # Context metrics (if ground truth available)
        if relevant_contexts:
            results['context_precision'] = RAGASMetrics.context_precision(contexts, relevant_contexts)
            results['context_recall'] = RAGASMetrics.context_recall(contexts, relevant_contexts)
        
        return results


class BenchmarkEvaluator:
    """Evaluate the system on benchmark datasets"""
    
    def __init__(self, retrieval_system, rag_system):
        self.retrieval_system = retrieval_system
        self.rag_system = rag_system
    
    def evaluate_beir_style(
        self,
        queries: List[Dict],
        corpus: List[Dict],
        qrels: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """
        Evaluate on BEIR-style benchmark
        
        Args:
            queries: List of query dicts with 'id' and 'text'
            corpus: List of document dicts with 'id' and 'text'
            qrels: Query relevance judgments {query_id: {doc_id: relevance_score}}
        """
        all_metrics = []
        
        for query in queries:
            query_id = query['id']
            query_text = query['text']
            
            # Get ground truth
            if query_id not in qrels:
                continue
            
            relevant_ids = list(qrels[query_id].keys())
            relevance_scores = qrels[query_id]
            
            # Run retrieval
            results = self.retrieval_system.search(query_text, top_k=100)
            retrieved_ids = [doc['id'] for doc, _ in results]
            
            # Evaluate
            metrics = IRMetrics.evaluate_retrieval(
                retrieved_ids,
                relevant_ids,
                relevance_scores
            )
            
            all_metrics.append(metrics)
        
        # Average across queries
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def evaluate_rag_end_to_end(
        self,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate RAG system end-to-end
        
        Args:
            test_cases: List of dicts with 'query', 'contexts', 'expected_answer', etc.
        """
        all_metrics = []
        
        for case in test_cases:
            query = case['query']
            contexts = case.get('contexts', [])
            relevant_contexts = case.get('relevant_contexts', None)
            
            # Run RAG
            result = self.rag_system.run(query)
            answer = result.get('synthesis', '')
            retrieved_contexts = [p.get('abstract', '') for p in result.get('reranked_papers', [])]
            
            # Evaluate
            metrics = RAGASMetrics.evaluate_rag(
                query,
                answer,
                retrieved_contexts,
                relevant_contexts
            )
            
            all_metrics.append(metrics)
        
        # Average across cases
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


def print_evaluation_report(metrics: Dict[str, float]):
    """Pretty print evaluation metrics"""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    for metric_name, value in sorted(metrics.items()):
        print(f"{metric_name:.<30} {value:.4f}")
    
    print("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    # Example: Evaluate retrieval
    retrieved_ids = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant_ids = ['doc2', 'doc4', 'doc6']
    relevance_scores = {'doc1': 0, 'doc2': 3, 'doc3': 0, 'doc4': 2, 'doc5': 0, 'doc6': 3}
    
    metrics = IRMetrics.evaluate_retrieval(retrieved_ids, relevant_ids, relevance_scores)
    print_evaluation_report(metrics)
    
    # Example: Evaluate RAG
    question = "What is retrieval augmented generation?"
    answer = "Retrieval augmented generation (RAG) is a technique that combines retrieval with generation."
    contexts = [
        "RAG combines retrieval with language models.",
        "The technique improves factual accuracy."
    ]
    
    rag_metrics = RAGASMetrics.evaluate_rag(question, answer, contexts)
    print_evaluation_report(rag_metrics)
