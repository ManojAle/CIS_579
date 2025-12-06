import streamlit as st
from agent import ResearchAgent
from enhanced_agent import EnhancedResearchAgent
from config import Config
import logging
import uuid

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
if 'history' not in st.session_state:
    st.session_state.history = []

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Header
st.markdown('<h1 class="main-header">🔬 Research Assistant AI</h1>', unsafe_allow_html=True)
st.markdown("**Intelligent literature discovery, comparison, and synthesis powered by Claude and hybrid retrieval**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode Selection
    st.subheader("🎯 Mode Selection")
    mode = st.radio(
        "Choose Mode",
        ["⚡ Standard (Fast)", "🚀 Enhanced (Features)"],
        help="Standard: Fast, direct search\nEnhanced: +Wikipedia, +Quality checks, +Memory"
    )
    
    use_enhanced = (mode == "🚀 Enhanced (Features)")
    
    if use_enhanced:
        use_memory = st.checkbox(
            "💾 Enable Memory", 
            value=False,
            help="Remember conversation context across queries"
        )
        
        st.info("""
        **Enhanced features:**
        - 🔧 Wikipedia context
        - 🔀 Smart query routing
        - ✅ Quality validation
        - 💾 Conversation memory
        """)
    
    # Data sources
    st.subheader("Data Sources")
    use_arxiv = st.checkbox("arXiv", value=True)
    use_semantic = st.checkbox("Semantic Scholar", value=False, disabled=True,
                               help="Currently unavailable (403 Forbidden)")
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top K Results", 5, 20, Config.TOP_K_RERANK)
    alpha = st.slider("BM25 Weight (α)", 0.0, 1.0, 0.5, 0.1,
                     help="0 = Dense only, 1 = BM25 only, 0.5 = Equal weight")
    
    # About
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This Research Assistant uses:
    - **Hybrid Retrieval**: BM25 + Dense Embeddings
    - **Re-ranking**: Cross-encoder scoring
    - **LangGraph**: Agentic workflow
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
        st.write("")
        st.write("")
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
                # Choose agent based on mode
                if use_enhanced:
                    if use_memory:
                        agent = EnhancedResearchAgent(use_memory=True)
                        config = {"configurable": {"thread_id": st.session_state.session_id}}
                        result = agent.run(query=query, user_context=context if context else "", config=config)
                    else:
                        agent = EnhancedResearchAgent(use_memory=False)
                        result = agent.run(query=query, user_context=context if context else "")
                    
                    # Show enhanced features used
                    enhanced_info = []
                    if result.get('tool_calls'):
                        enhanced_info.append(f"🔧 {len(result['tool_calls'])} tools used")
                    
                    classification = [m for m in result.get('messages', []) if 'classified' in m.lower()]
                    if classification:
                        enhanced_info.append(f"📊 {classification[0]}")
                    
                    if enhanced_info:
                        st.info(" | ".join(enhanced_info))
                else:
                    # Original agent
                    agent = ResearchAgent()
                    result = agent.run(query=query, user_context=context if context else "")
                
                # Check for errors
                if result.get('error'):
                    st.error(f"❌ Error: {result['error']}")
                else:
                    # Save to history
                    st.session_state.history.append({
                        'query': query,
                        'result': result,
                        'enhanced': use_enhanced
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
                        if use_enhanced and 'quality_score' in result:
                            st.metric("Quality Score", f"{result['quality_score']:.2f}")
                        else:
                            st.metric("Steps", result.get('step_count', 0))
                    
                    st.divider()
                    
                    # Show tool results if available
                    if use_enhanced and result.get('tool_calls'):
                        with st.expander("🔧 Background Context (from Wikipedia & ArXiv tools)"):
                            for i, tool_result in enumerate(result['tool_calls'], 1):
                                st.markdown(f"**Tool {i}:**")
                                st.write(tool_result[:400] + "...")
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
            mode_badge = "🚀 Enhanced" if item.get('enhanced') else "⚡ Standard"
            with st.expander(f"{mode_badge} - Query {len(st.session_state.history) - i + 1}: {item['query']}"):
                st.markdown("**Synthesis:**")
                st.write(item['result'].get('synthesis', 'N/A')[:500] + "...")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Papers", len(item['result'].get('reranked_papers', [])))
                with col2:
                    if item.get('enhanced') and 'quality_score' in item['result']:
                        st.metric("Quality", f"{item['result']['quality_score']:.2f}")
                with col3:
                    if item.get('enhanced') and item['result'].get('tool_calls'):
                        st.metric("Tools", len(item['result']['tool_calls']))
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

with tab3:
    st.subheader("ℹ️ How to Use")
    
    st.markdown("""
    ### Mode Selection
    
    **⚡ Standard Mode (Fast)**
    - Direct paper search
    - BM25 + Dense retrieval
    - ~15-20 seconds
    - Best for: Quick lookups
    
    **🚀 Enhanced Mode (Features)**
    - + Wikipedia background
    - + Smart query routing
    - + Quality validation
    - + Memory (optional)
    - ~20-28 seconds
    - Best for: Complex research, conversations
    
    ### Getting Started
    
    1. **Choose your mode** in the sidebar
    2. **Enter your research query**
    3. **Optionally add context** for better results
    4. **Click Search** and wait for results
    
    ### Enhanced Mode Features
    
    **Smart Routing:**
    - "Simple" queries → Direct search
    - "Complex" queries → Use Wikipedia first
    - "Unclear" queries → Ask for clarification
    
    **Quality Validation:**
    - Scores synthesis quality (0-1)
    - Automatic retry if needed
    - Ensures comprehensive results
    
    **Memory (Optional):**
    - Remembers conversation context
    - Better for follow-up questions
    - Uses session-based tracking
    
    ### Example Queries
    
    **Standard Mode:**
    ```
    "transformer attention mechanisms"
    "BERT architecture"
    "neural machine translation"
    ```
    
    **Enhanced Mode:**
    ```
    "quantum computing applications in ML"
    "history and future of federated learning"
    "compare RAG architectures"
    ```
    
    ### Tips
    
    - Use **specific queries** for better results
    - Add **context** to help the AI understand your needs
    - Try **Enhanced mode** for exploratory research
    - Use **Memory** for multi-turn conversations
    - Check **workflow steps** to see what happened
    """)
    
    st.divider()
    
    st.markdown("### 🛠️ System Configuration")
    st.code(f"""
Mode: {'Enhanced' if use_enhanced else 'Standard'}
Memory: {'Enabled' if (use_enhanced and use_memory) else 'Disabled'}
Embedding Model: {Config.EMBEDDING_MODEL}
LLM Model: {Config.LLM_MODEL}
Top-K Results: {top_k}
BM25 Weight: {alpha}
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Claude API, LangGraph, and Streamlit</p>
    <p>ECE 579 Intelligent Systems - Manoj Alexender & Seraj Aldwake</p>
</div>
""", unsafe_allow_html=True)
