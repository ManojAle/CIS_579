import streamlit as st
from agent import ResearchAgent
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Research Assistant AI",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .paper-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .citation {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    try:
        Config.validate()
        st.session_state.agent = ResearchAgent()
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.init_error = str(e)

if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown('<h1 class="main-header">🔬 Research Assistant AI</h1>', unsafe_allow_html=True)
st.markdown("**Intelligent literature discovery, comparison, and synthesis powered by Claude and hybrid retrieval**")

# Check initialization
if not st.session_state.get('initialized', False):
    st.error(f"❌ Initialization Error: {st.session_state.get('init_error', 'Unknown error')}")
    st.info("Please ensure your `.env` file is configured with a valid `ANTHROPIC_API_KEY`")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Data sources
    st.subheader("Data Sources")
    use_arxiv = st.checkbox("arXiv", value=True)
    use_semantic = st.checkbox("Semantic Scholar", value=True)
    use_google = st.checkbox("Google Scholar", value=bool(Config.SERPAPI_KEY), 
                            disabled=not Config.SERPAPI_KEY)
    
    if not Config.SERPAPI_KEY and use_google:
        st.info("💡 Add SERPAPI_KEY to .env to enable Google Scholar")
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top K Results", 5, 20, Config.TOP_K_RERANK)
    alpha = st.slider("BM25 Weight (α)", 0.0, 1.0, 0.5, 0.1,
                     help="0 = Dense only, 1 = BM25 only, 0.5 = Equal weight")
    
    # arXiv categories
    st.subheader("arXiv Categories")
    categories = st.multiselect(
        "Filter by categories",
        options=Config.ARXIV_CATEGORIES,
        default=["cs.AI", "cs.CL", "cs.LG"]
    )
    
    # About
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This Research Assistant uses:
    - **Hybrid Retrieval**: BM25 + Dense Embeddings
    - **Re-ranking**: Cross-encoder scoring
    - **LangGraph**: Agentic workflow orchestration
    - **Claude API**: Analysis & synthesis
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📚 History", "ℹ️ Help"])

with tab1:
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Research Query",
            placeholder="e.g., retrieval augmented generation in conversational AI",
            help="Enter your research question or topic"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Optional context
    with st.expander("📝 Add Context (Optional)"):
        context = st.text_area(
            "Provide additional context for your research",
            placeholder="e.g., I'm working on a chatbot project and need to understand RAG architectures...",
            height=100
        )
    
    # Process search
    if search_button and query:
        with st.spinner("🔄 Searching and analyzing papers..."):
            try:
                # Prepare sources list
                sources = []
                if use_arxiv:
                    sources.append('arxiv')
                if use_semantic:
                    sources.append('semantic_scholar')
                if use_google and Config.SERPAPI_KEY:
                    sources.append('google_scholar')
                
                # Run agent
                result = st.session_state.agent.run(
                    query=query,
                    user_context=context if context else ""
                )
                
                # Check for errors
                if result.get('error'):
                    st.error(f"❌ Error: {result['error']}")
                else:
                    # Save to history
                    st.session_state.history.append({
                        'query': query,
                        'result': result
                    })
                    
                    # Display results
                    st.success("✅ Analysis complete!")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Papers Found", len(result.get('raw_papers', [])))
                    with col2:
                        st.metric("Retrieved", len(result.get('retrieved_papers', [])))
                    with col3:
                        st.metric("Final Selection", len(result.get('reranked_papers', [])))
                    with col4:
                        st.metric("Steps", result.get('step_count', 0))
                    
                    st.divider()
                    
                    # Synthesis
                    st.subheader("📊 Synthesis")
                    st.markdown(result.get('synthesis', 'No synthesis available'))
                    
                    st.divider()
                    
                    # Top papers
                    st.subheader("📄 Top Relevant Papers")
                    
                    papers = result.get('reranked_papers', [])
                    for i, paper in enumerate(papers[:5], 1):
                        with st.expander(f"**{i}. {paper.get('title', 'Unknown')}**"):
                            authors = ', '.join(paper.get('authors', [])[:3])
                            if len(paper.get('authors', [])) > 3:
                                authors += ' et al.'
                            
                            year = paper.get('year', paper.get('published', 'N/A'))
                            venue = paper.get('venue', paper.get('source', 'N/A'))
                            
                            st.markdown(f"**Authors:** {authors}")
                            st.markdown(f"**Year:** {year} | **Venue:** {venue}")
                            
                            if paper.get('citation_count'):
                                st.markdown(f"**Citations:** {paper.get('citation_count')}")
                            
                            st.markdown("**Abstract:**")
                            st.write(paper.get('abstract', 'No abstract available'))
                            
                            # Links
                            links = []
                            if paper.get('pdf_url'):
                                links.append(f"[PDF]({paper.get('pdf_url')})")
                            if paper.get('url'):
                                links.append(f"[URL]({paper.get('url')})")
                            
                            if links:
                                st.markdown(" | ".join(links))
                    
                    st.divider()
                    
                    # Workflow messages
                    with st.expander("🔍 View Workflow Steps"):
                        for msg in result.get('messages', []):
                            st.info(msg)
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                logger.error(f"Search error: {e}", exc_info=True)

with tab2:
    st.subheader("📚 Search History")
    
    if not st.session_state.history:
        st.info("No search history yet. Start by making a query!")
    else:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Query {len(st.session_state.history) - i + 1}: {item['query']}"):
                st.markdown("**Synthesis:**")
                st.write(item['result'].get('synthesis', 'N/A'))
                
                st.markdown(f"**Papers Found:** {len(item['result'].get('reranked_papers', []))}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

with tab3:
    st.subheader("ℹ️ How to Use")
    
    st.markdown("""
    ### Getting Started
    
    1. **Enter your research query** in the search box
    2. **Optionally add context** to provide more information about your research needs
    3. **Click Search** to start the analysis
    
    ### Features
    
    - **Multi-source Search**: Searches arXiv, Semantic Scholar, and optionally Google Scholar
    - **Hybrid Retrieval**: Combines keyword matching (BM25) with semantic search
    - **Re-ranking**: Uses cross-encoders to improve relevance
    - **AI Analysis**: Claude analyzes papers and generates insights
    - **Citation Tracking**: Proper citations for all referenced papers
    
    ### Tips
    
    - Use specific, focused queries for best results
    - Add context to help the AI understand your research goals
    - Adjust the BM25 weight (α) to favor keyword vs semantic matching
    - Enable multiple data sources for comprehensive coverage
    
    ### System Architecture
    
    ```
    Query → Search → Hybrid Retrieval → Re-rank → Analyze → Synthesize → Results
    ```
    
    ### Technologies Used
    
    - **LangGraph**: Orchestrates the multi-step workflow
    - **Claude API**: Powers analysis and synthesis
    - **SPECTER2**: Scientific paper embeddings
    - **BM25**: Lexical retrieval
    - **Cross-Encoder**: Re-ranking
    """)
    
    st.divider()
    
    st.markdown("### 🛠️ Configuration")
    st.code(f"""
Model: {Config.LLM_MODEL}
Embedding Model: {Config.EMBEDDING_MODEL}
Max Results: {Config.MAX_RESULTS}
Top-K Retrieval: {Config.TOP_K_RETRIEVAL}
Top-K Rerank: {Config.TOP_K_RERANK}
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Claude API, LangGraph, and Streamlit</p>
    <p>ECE 579 Intelligent Systems - Manoj Alexender & Seraj Aldwake</p>
</div>
""", unsafe_allow_html=True)
