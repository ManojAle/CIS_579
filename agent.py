from typing import TypedDict, List, Dict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import logging

from config import Config
from retrieval import UnifiedRetriever
from hybrid_retrieval import HybridRetriever, Reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    """State for the research agent workflow"""
    query: str
    user_context: str
    raw_papers: List[Dict]
    retrieved_papers: List[Dict]
    reranked_papers: List[Dict]
    analysis: str
    synthesis: str
    citations: List[str]
    error: str
    step_count: int
    messages: Annotated[List, operator.add]


class ResearchAgent:
    """Agentic workflow for research assistance using LangGraph"""
    
    def __init__(self):
        Config.validate()
        
        # Initialize components
        self.llm = ChatAnthropic(
            model=Config.LLM_MODEL,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        self.retriever = UnifiedRetriever()
        self.hybrid_retriever = HybridRetriever(Config.EMBEDDING_MODEL)
        self.reranker = Reranker()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("search", self.search_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("synthesize", self.synthesize_node)
        
        # Add edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "analyze")
        workflow.add_edge("analyze", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def search_node(self, state: ResearchState) -> ResearchState:
        """Node 1: Search for papers across data sources"""
        logger.info(f"Searching for papers: {state['query']}")
        
        try:
            papers = self.retriever.search_all(
                state['query'],
                sources=['arxiv', 'semantic_scholar']
            )
            
            state['raw_papers'] = papers
            state['step_count'] = state.get('step_count', 0) + 1
            state['messages'] = [f"Found {len(papers)} papers from multiple sources"]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            state['error'] = str(e)
            state['messages'] = [f"Error in search: {e}"]
        
        return state
    
    def retrieve_node(self, state: ResearchState) -> ResearchState:
        """Node 2: Hybrid retrieval (BM25 + Dense)"""
        logger.info("Performing hybrid retrieval...")
        
        try:
            papers = state['raw_papers']
            
            if not papers:
                state['retrieved_papers'] = []
                state['messages'].append("No papers to retrieve from")
                return state
            
            # Index papers
            self.hybrid_retriever.index(papers)
            
            # Perform hybrid search
            results = self.hybrid_retriever.search(
                state['query'],
                top_k=Config.TOP_K_RETRIEVAL,
                alpha=0.5  # Equal weight to BM25 and dense
            )
            
            state['retrieved_papers'] = [doc for doc, score in results]
            state['step_count'] += 1
            state['messages'].append(f"Retrieved top {len(results)} papers using hybrid search")
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            state['error'] = str(e)
            state['messages'].append(f"Error in retrieval: {e}")
        
        return state
    
    def rerank_node(self, state: ResearchState) -> ResearchState:
        """Node 3: Re-rank retrieved papers"""
        logger.info("Re-ranking papers...")
        
        try:
            retrieved = [(doc, 0.0) for doc in state['retrieved_papers']]
            
            if not retrieved:
                state['reranked_papers'] = []
                state['messages'].append("No papers to rerank")
                return state
            
            # Rerank using cross-encoder
            reranked = self.reranker.rerank(
                state['query'],
                retrieved,
                top_k=Config.TOP_K_RERANK
            )
            
            state['reranked_papers'] = [doc for doc, score in reranked]
            state['step_count'] += 1
            state['messages'].append(f"Reranked to top {len(reranked)} papers")
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            state['error'] = str(e)
            state['messages'].append(f"Error in reranking: {e}")
        
        return state
    
    def analyze_node(self, state: ResearchState) -> ResearchState:
        """Node 4: Analyze papers and extract insights"""
        logger.info("Analyzing papers...")
        
        try:
            papers = state['reranked_papers']
            
            if not papers:
                state['analysis'] = "No papers available for analysis."
                return state
            
            # Prepare context for LLM
            papers_context = self._format_papers_for_llm(papers)
            
            system_prompt = """You are a research assistant analyzing scientific papers.
Your task is to analyze the provided papers and extract:
1. Key themes and topics
2. Main methodologies used
3. Important findings
4. Research gaps identified
5. Citation relationships (who cites whom)

Be concise and factual. Focus on actionable insights."""
            
            user_prompt = f"""Query: {state['query']}

User Context: {state.get('user_context', 'General research inquiry')}

Papers to analyze:
{papers_context}

Provide a structured analysis of these papers."""
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            state['analysis'] = response.content
            state['step_count'] += 1
            state['messages'].append("Completed paper analysis")
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            state['error'] = str(e)
            state['messages'].append(f"Error in analysis: {e}")
        
        return state
    
    def synthesize_node(self, state: ResearchState) -> ResearchState:
        """Node 5: Synthesize findings and generate response"""
        logger.info("Synthesizing final response...")
        
        try:
            papers = state['reranked_papers']
            analysis = state.get('analysis', '')
            
            # Prepare synthesis prompt
            system_prompt = """You are a research assistant providing synthesized insights.

Generate a comprehensive response that includes:
1. Direct answer to the user's query
2. Key findings from the literature
3. Comparative analysis (if applicable)
4. Research gaps and future directions
5. Properly formatted citations

Format citations as [Author et al., Year] and include a references section at the end.
Be clear, concise, and grounded in the provided papers."""
            
            papers_context = self._format_papers_for_llm(papers)
            
            user_prompt = f"""Query: {state['query']}

User Context: {state.get('user_context', 'General research inquiry')}

Analysis:
{analysis}

Papers:
{papers_context}

Synthesize a comprehensive response with proper citations."""
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            state['synthesis'] = response.content
            state['citations'] = self._extract_citations(papers)
            state['step_count'] += 1
            state['messages'].append("Completed synthesis")
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            state['error'] = str(e)
            state['messages'].append(f"Error in synthesis: {e}")
        
        return state
    
    def _format_papers_for_llm(self, papers: List[Dict], max_papers: int = 10) -> str:
        """Format papers for LLM context"""
        formatted = []
        
        for i, paper in enumerate(papers[:max_papers], 1):
            title = paper.get('title', 'Unknown')
            authors = ', '.join(paper.get('authors', [])[:3])
            if len(paper.get('authors', [])) > 3:
                authors += ' et al.'
            
            year = paper.get('year', paper.get('published', 'Unknown'))
            abstract = paper.get('abstract', 'No abstract available')[:500]
            
            formatted.append(f"""
Paper {i}:
Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}...
""")
        
        return "\n".join(formatted)
    
    def _extract_citations(self, papers: List[Dict]) -> List[str]:
        """Extract formatted citations from papers"""
        citations = []
        
        for paper in papers:
            authors = paper.get('authors', [])
            if authors:
                author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
            else:
                author_str = "Unknown"
            
            year = paper.get('year', paper.get('published', 'n.d.'))
            title = paper.get('title', 'Untitled')
            
            citation = f"{author_str} ({year}). {title}"
            citations.append(citation)
        
        return citations
    
    def run(self, query: str, user_context: str = "") -> Dict:
        """Run the research agent workflow"""
        initial_state = ResearchState(
            query=query,
            user_context=user_context,
            raw_papers=[],
            retrieved_papers=[],
            reranked_papers=[],
            analysis="",
            synthesis="",
            citations=[],
            error="",
            step_count=0,
            messages=[]
        )
        
        logger.info(f"Starting research workflow for query: {query}")
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {
                'error': str(e),
                'synthesis': f"An error occurred during research: {e}"
            }


# Simple API wrapper
def research_query(query: str, context: str = "") -> Dict:
    """Simple function to query the research assistant"""
    agent = ResearchAgent()
    return agent.run(query, context)
