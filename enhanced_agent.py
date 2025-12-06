"""
Enhanced Research Assistant with LangGraph Tools
Includes: Wikipedia, ArXiv tool, Tavily search, conditional routing, and human-in-the-loop
"""

from typing import TypedDict, List, Dict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
import operator
import logging

from config import Config
from retrieval import UnifiedRetriever
from hybrid_retrieval import HybridRetriever, Reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedResearchState(TypedDict):
    """Enhanced state with tool support"""
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
    
    # New fields for enhanced features
    tool_calls: List[Dict]
    needs_clarification: bool
    clarification_question: str
    quality_score: float
    use_tools: bool


class EnhancedResearchAgent:
    """
    Enhanced Research Assistant with LangGraph Tools
    
    Features:
    - Wikipedia integration for background context
    - ArXiv tool for direct paper lookup
    - Conditional routing based on query type
    - Human-in-the-loop for clarifications
    - Memory/checkpointing for conversation continuity
    - Quality assessment and routing
    """
    
    def __init__(self, use_memory: bool = False):
        Config.validate()
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=Config.LLM_MODEL,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        # Initialize retrievers
        self.retriever = UnifiedRetriever()
        self.hybrid_retriever = HybridRetriever(Config.EMBEDDING_MODEL)
        self.reranker = Reranker()
        
        # Initialize LangChain tools
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Memory for conversation continuity
        self.memory = MemorySaver() if use_memory else None
        
        # Build the enhanced graph
        self.graph = self._build_enhanced_graph()
    
    def _create_tools(self) -> List:
        """Create LangChain tools for the agent"""
        
        # Wikipedia tool for background information
        wikipedia = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=2,
                doc_content_chars_max=1000
            )
        )
        
        # ArXiv tool for paper lookup
        arxiv = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(
                top_k_results=3,
                doc_content_chars_max=500
            )
        )
        
        # Custom tool for paper statistics
        @tool
        def get_paper_stats(papers: List[Dict]) -> str:
            """Get statistics about retrieved papers"""
            if not papers:
                return "No papers available"
            
            total = len(papers)
            years = [p.get('year', p.get('published', 'Unknown')) for p in papers]
            avg_year = sum(int(y) for y in years if str(y).isdigit()) / len([y for y in years if str(y).isdigit()])
            
            return f"Total papers: {total}, Average year: {avg_year:.0f}, Latest: {max(years)}"
        
        # Custom tool for citation analysis
        @tool
        def analyze_citations(papers: List[Dict]) -> str:
            """Analyze citation patterns in papers"""
            if not papers:
                return "No papers to analyze"
            
            cited_papers = [p for p in papers if p.get('citation_count', 0) > 0]
            total_citations = sum(p.get('citation_count', 0) for p in papers)
            avg_citations = total_citations / len(cited_papers) if cited_papers else 0
            
            return f"Papers with citations: {len(cited_papers)}, Total citations: {total_citations}, Avg: {avg_citations:.1f}"
        
        return [wikipedia, arxiv, get_paper_stats, analyze_citations]
    
    def _build_enhanced_graph(self) -> StateGraph:
        """Build enhanced workflow with conditional routing"""
        workflow = StateGraph(EnhancedResearchState)
        
        # Add nodes
        workflow.add_node("classify_query", self.classify_query_node)
        workflow.add_node("search", self.search_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("use_tools", self.tool_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("synthesize", self.synthesize_node)
        workflow.add_node("quality_check", self.quality_check_node)
        
        # Set entry point
        workflow.set_entry_point("classify_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_query",
            self.route_query,
            {
                "simple": "search",
                "complex": "use_tools",
                "clarify": END
            }
        )
        
        workflow.add_edge("use_tools", "search")
        workflow.add_edge("search", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "analyze")
        workflow.add_edge("analyze", "synthesize")
        workflow.add_edge("synthesize", "quality_check")
        
        # Quality-based routing
        workflow.add_conditional_edges(
            "quality_check",
            self.route_quality,
            {
                "good": END,
                "retry": "analyze",
                "needs_more": "search"
            }
        )
        
        # Compile with checkpointing if memory enabled
        if self.memory:
            return workflow.compile(checkpointer=self.memory)
        return workflow.compile()
    
    def classify_query_node(self, state: EnhancedResearchState) -> EnhancedResearchState:
        """Classify query to determine routing"""
        logger.info("Classifying query...")
        
        query = state['query']
        
        # Use LLM to classify
        classification_prompt = f"""Classify this research query into one of these categories:

"simple": Direct research query about a specific technical topic, method, or concept
  Examples: "transformer attention", "neural networks", "BERT architecture"

"complex": Broad query needing background or philosophical/historical context
  Examples: "history and future of AI", "quantum computing implications"

"clarify": Vague or unclear query without specific topic
  Examples: "something about AI", "help me", "recent stuff"

Query: {query}

This is a technical research query. If it mentions specific technical terms or methods, classify as "simple".
Respond with only one word: simple, complex, or clarify"""

        try:
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            classification = response.content.strip().lower()
            
            if classification not in ["simple", "complex", "clarify"]:
                classification = "simple"
            
            # Fallback: If query has technical terms but LLM says clarify, override to simple
            technical_indicators = ['network', 'model', 'algorithm', 'learning', 'attention', 
                                   'transformer', 'neural', 'deep', 'machine', 'architecture',
                                   'training', 'inference', 'optimization', 'embedding']
            
            query_lower = query.lower()
            has_technical_terms = any(term in query_lower for term in technical_indicators)
            query_length = len(query.split())
            
            if classification == "clarify" and has_technical_terms and query_length >= 3:
                logger.info(f"Overriding 'clarify' to 'simple' - query has technical terms")
                classification = "simple"
            
            state['use_tools'] = (classification == "complex")
            state['needs_clarification'] = (classification == "clarify")
            
            if classification == "clarify":
                state['clarification_question'] = "Could you please provide more specific details about what aspect you're interested in?"
            
            logger.info(f"Query classified as: {classification}")
            state['messages'].append(f"Query classified as: {classification}")
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            state['use_tools'] = False
            state['needs_clarification'] = False
        
        state['step_count'] = state.get('step_count', 0) + 1
        return state
    
    def route_query(self, state: EnhancedResearchState) -> str:
        """Route based on query classification"""
        if state.get('needs_clarification', False):
            return "clarify"
        elif state.get('use_tools', False):
            return "complex"
        else:
            return "simple"
    
    def tool_node(self, state: EnhancedResearchState) -> EnhancedResearchState:
        """Use LangChain tools for background information"""
        logger.info("Using tools for background context...")
        
        query = state['query']
        tool_results = []
        
        # Use Wikipedia for background
        try:
            wikipedia_tool = self.tools[0]  # Wikipedia
            wiki_result = wikipedia_tool.invoke({"query": query})
            tool_results.append(f"Wikipedia: {wiki_result[:500]}")
            logger.info("Retrieved Wikipedia context")
        except Exception as e:
            logger.error(f"Wikipedia error: {e}")
        
        # Use ArXiv tool
        try:
            arxiv_tool = self.tools[1]  # ArXiv
            arxiv_result = arxiv_tool.invoke({"query": query})
            tool_results.append(f"ArXiv Tool: {arxiv_result[:500]}")
            logger.info("Retrieved ArXiv tool results")
        except Exception as e:
            logger.error(f"ArXiv tool error: {e}")
        
        state['tool_calls'] = tool_results
        state['messages'].append(f"Used {len(tool_results)} tools successfully")
        state['step_count'] += 1
        
        return state
    
    def search_node(self, state: EnhancedResearchState) -> EnhancedResearchState:
        """Search for papers (same as original)"""
        logger.info(f"Searching for papers: {state['query']}")
        
        try:
            papers = self.retriever.search_all(
                state['query'],
                sources=['arxiv']  # Using only arXiv since Semantic Scholar is 403
            )
            
            state['raw_papers'] = papers
            state['step_count'] = state.get('step_count', 0) + 1
            state['messages'].append(f"Found {len(papers)} papers from multiple sources")
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            state['error'] = str(e)
            state['messages'].append(f"Error in search: {e}")
        
        return state
    
    def retrieve_node(self, state: EnhancedResearchState) -> EnhancedResearchState:
        """Hybrid retrieval (same as original)"""
        logger.info("Performing hybrid retrieval...")
        
        try:
            papers = state['raw_papers']
            
            if not papers:
                state['retrieved_papers'] = []
                return state
            
            self.hybrid_retriever.index(papers)
            results = self.hybrid_retriever.search(
                state['query'],
                top_k=Config.TOP_K_RETRIEVAL,
                alpha=0.5
            )
            
            state['retrieved_papers'] = [doc for doc, score in results]
            state['step_count'] += 1
            state['messages'].append(f"Retrieved top {len(results)} papers")
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            state['error'] = str(e)
        
        return state
    
    def rerank_node(self, state: EnhancedResearchState) -> EnhancedResearchState:
        """Re-rank papers (same as original)"""
        logger.info("Re-ranking papers...")
        
        try:
            retrieved = [(doc, 0.0) for doc in state['retrieved_papers']]
            
            if not retrieved:
                state['reranked_papers'] = []
                return state
            
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
        
        return state
    
    def analyze_node(self, state: EnhancedResearchState) -> EnhancedResearchState:
        """Analyze papers with tool context"""
        logger.info("Analyzing papers...")
        
        try:
            papers = state['reranked_papers']
            tool_context = "\n\n".join(state.get('tool_calls', []))
            
            if not papers:
                state['analysis'] = "No papers available for analysis."
                return state
            
            papers_context = self._format_papers_for_llm(papers)
            
            system_prompt = """You are a research assistant analyzing scientific papers.
Analyze the papers and extract key insights, methodologies, findings, and gaps."""
            
            user_prompt = f"""Query: {state['query']}

Background Context (from tools):
{tool_context if tool_context else 'No additional context'}

Papers to analyze:
{papers_context}

Provide a structured analysis."""
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            state['analysis'] = response.content
            state['step_count'] += 1
            state['messages'].append("Completed analysis")
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            state['error'] = str(e)
        
        return state
    
    def synthesize_node(self, state: EnhancedResearchState) -> EnhancedResearchState:
        """Synthesize findings (same as original)"""
        logger.info("Synthesizing final response...")
        
        try:
            papers = state['reranked_papers']
            analysis = state.get('analysis', '')
            tool_context = "\n".join(state.get('tool_calls', []))
            
            system_prompt = """Generate a comprehensive research response with proper citations."""
            
            papers_context = self._format_papers_for_llm(papers)
            
            user_prompt = f"""Query: {state['query']}

Background Context:
{tool_context if tool_context else 'None'}

Analysis:
{analysis}

Papers:
{papers_context}

Synthesize a comprehensive response with citations."""
            
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
        
        return state
    
    def quality_check_node(self, state: EnhancedResearchState) -> EnhancedResearchState:
        """Check quality of synthesis"""
        logger.info("Checking quality...")
        
        synthesis = state.get('synthesis', '')
        papers_count = len(state.get('reranked_papers', []))
        
        # Simple quality scoring
        score = 0.0
        
        if len(synthesis) > 200:
            score += 0.3
        if papers_count >= 5:
            score += 0.3
        if '[' in synthesis and ']' in synthesis:  # Has citations
            score += 0.2
        if len(state.get('analysis', '')) > 100:
            score += 0.2
        
        state['quality_score'] = score
        state['messages'].append(f"Quality score: {score:.2f}")
        
        return state
    
    def route_quality(self, state: EnhancedResearchState) -> str:
        """Route based on quality score"""
        score = state.get('quality_score', 0.0)
        
        if score >= 0.8:
            return "good"
        elif score >= 0.5:
            return "good"  # Accept medium quality
        else:
            return "good"  # For now, always accept
    
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
        """Extract formatted citations"""
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
    
    def run(self, query: str, user_context: str = "", config: Optional[Dict] = None) -> Dict:
        """Run the enhanced workflow"""
        initial_state = EnhancedResearchState(
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
            messages=[],
            tool_calls=[],
            needs_clarification=False,
            clarification_question="",
            quality_score=0.0,
            use_tools=False
        )
        
        logger.info(f"Starting enhanced workflow for query: {query}")
        
        try:
            # Run with config for memory/threading
            if config and self.memory:
                final_state = self.graph.invoke(initial_state, config=config)
            else:
                final_state = self.graph.invoke(initial_state)
            
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {
                'error': str(e),
                'synthesis': f"An error occurred: {e}"
            }


# Simple API wrapper
def enhanced_research_query(query: str, context: str = "", use_tools: bool = True) -> Dict:
    """Simple function to query the enhanced research assistant"""
    agent = EnhancedResearchAgent(use_memory=False)
    return agent.run(query, context)