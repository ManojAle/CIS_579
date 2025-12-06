# Research Assistant AI 🔬

An intelligent research assistant that discovers, retrieves, compares, and synthesizes academic literature using hybrid retrieval (BM25 + Dense Embeddings), re-ranking, and Claude API for analysis.

**Authors:** Manoj Alexender & Seraj Aldwake  
**Course:** ECE 579 Intelligent Systems  
**Institution:** [Your University]

## 🌟 Features

- **Multi-Source Search**: Integrates arXiv, Semantic Scholar, and Google Scholar APIs
- **Hybrid Retrieval**: Combines BM25 (lexical) with SPECTER2 embeddings (semantic)
- **Intelligent Re-ranking**: Uses cross-encoders for precision optimization
- **Agentic Workflows**: LangGraph orchestrates multi-step research tasks
- **AI-Powered Analysis**: Claude API generates grounded insights, comparisons, and summaries
- **Citation Management**: Automatic citation formatting and tracking
- **Interactive UI**: Streamlit web interface with search history
- **Comprehensive Evaluation**: BEIR and RAGAS metrics for quality assessment

## 🏗️ Architecture

```
Query → Multi-Source Search → Hybrid Retrieval → Re-ranking → Analysis → Synthesis
         (arXiv, S2, Scholar)   (BM25 + Dense)   (Cross-Enc)  (Claude)   (Claude)
```

### System Components

1. **Data Retrieval Layer** (`retrieval.py`)
   - arXiv API integration
   - Semantic Scholar API integration
   - Google Scholar via SerpAPI (optional)
   - Unified retrieval interface with deduplication

2. **Hybrid Retrieval Engine** (`hybrid_retrieval.py`)
   - BM25 lexical retrieval
   - Dense retrieval with SPECTER2 embeddings
   - Score fusion (weighted combination)
   - Cross-encoder re-ranking

3. **Agentic Workflow** (`agent.py`)
   - LangGraph state machine
   - Multi-step orchestration (search → retrieve → rerank → analyze → synthesize)
   - Error handling and retry logic
   - Claude API integration for analysis

4. **Web Interface** (`app.py`)
   - Streamlit-based UI
   - Real-time search and analysis
   - Search history management
   - Interactive paper exploration

5. **Evaluation Suite** (`evaluation.py`)
   - IR metrics: Recall@k, Precision@k, nDCG, MRR
   - RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision/Recall
   - Benchmark support (BEIR-style)

## 📋 Requirements

- Python 3.8+
- Anthropic API key (required)
- SerpAPI key (optional, for Google Scholar)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd research-assistant-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.template .env
```

Edit `.env` and add your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
SERPAPI_KEY=your_serpapi_key_here  # Optional
SEMANTIC_SCHOLAR_API_KEY=          # Optional (for higher rate limits)

# Configuration
MAX_RESULTS=10
EMBEDDING_MODEL=sentence-transformers/allenai-specter2
LLM_MODEL=claude-sonnet-4-20250514
```

**Get API Keys:**
- **Anthropic API**: https://console.anthropic.com/
- **SerpAPI** (optional): https://serpapi.com/
- **Semantic Scholar** (optional): https://www.semanticscholar.org/product/api

## 💻 Usage

### Option 1: Web Interface (Recommended)

Launch the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Interactive search interface
- Adjustable retrieval parameters (BM25 weight, top-k)
- Data source selection
- Search history
- Detailed paper views

### Option 2: Command Line Interface

Run a quick search from the terminal:

```bash
python cli.py "retrieval augmented generation in conversational AI"
```

With additional context:

```bash
python cli.py "transformer attention mechanisms" --context "I'm building a chatbot"
```

Enable verbose logging:

```bash
python cli.py "few-shot learning" --verbose
```

### Option 3: Python API

Use the research assistant in your own Python code:

```python
from agent import research_query

# Simple query
result = research_query("neural machine translation")

# With context
result = research_query(
    query="neural machine translation",
    context="I'm interested in low-resource languages"
)

# Access results
print(result['synthesis'])  # AI-generated synthesis
print(result['reranked_papers'])  # Top relevant papers
print(result['citations'])  # Formatted citations
```

**Advanced Usage:**

```python
from agent import ResearchAgent
from config import Config

# Initialize agent
agent = ResearchAgent()

# Run with full control
result = agent.run(
    query="large language model reasoning",
    user_context="Focus on chain-of-thought prompting"
)

# Access intermediate results
print(f"Papers found: {len(result['raw_papers'])}")
print(f"Retrieved: {len(result['retrieved_papers'])}")
print(f"Final: {len(result['reranked_papers'])}")
print(f"Workflow steps: {result['step_count']}")
```

## 📊 Evaluation

Run evaluation on your retrieval system:

```python
from evaluation import IRMetrics, RAGASMetrics, print_evaluation_report

# Evaluate retrieval
retrieved_ids = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
relevant_ids = ['doc2', 'doc4', 'doc6']
relevance_scores = {
    'doc1': 0, 'doc2': 3, 'doc3': 0, 
    'doc4': 2, 'doc5': 0, 'doc6': 3
}

metrics = IRMetrics.evaluate_retrieval(
    retrieved_ids, 
    relevant_ids, 
    relevance_scores,
    k_values=[5, 10]
)

print_evaluation_report(metrics)
```

**Expected Output:**
```
============================================================
EVALUATION REPORT
============================================================
mrr................................ 0.5000
ndcg@5............................. 0.6131
ndcg@10............................ 0.6131
precision@5........................ 0.4000
precision@10....................... 0.2000
recall@5........................... 0.6667
recall@10.......................... 0.6667
============================================================
```

## 🔧 Configuration

Edit `config.py` or use environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `EMBEDDING_MODEL` | `allenai-specter2` | Embedding model for dense retrieval |
| `MAX_RESULTS` | 10 | Maximum results to return |
| `TOP_K_RETRIEVAL` | 20 | Documents to retrieve before re-ranking |
| `TOP_K_RERANK` | 10 | Final documents after re-ranking |
| `TEMPERATURE` | 0.3 | LLM temperature for generation |

## 📁 Project Structure

```
research-assistant-ai/
├── agent.py                 # LangGraph workflow and agent
├── app.py                   # Streamlit web interface
├── cli.py                   # Command-line interface
├── config.py                # Configuration management
├── evaluation.py            # Evaluation metrics (BEIR, RAGAS)
├── hybrid_retrieval.py      # Hybrid retrieval & re-ranking
├── retrieval.py             # Multi-source data retrieval
├── requirements.txt         # Python dependencies
├── .env.template            # Environment variables template
├── .env                     # Your API keys (not in git)
└── README.md               # This file
```

## 🧪 Example Queries

**Broad Research Questions:**
```
"What are the latest advances in multimodal learning?"
"Compare different approaches to few-shot learning"
"Research gaps in explainable AI"
```

**Specific Technical Questions:**
```
"How does SPECTER2 improve on SPECTER for paper embeddings?"
"Evaluation metrics for retrieval augmented generation systems"
"Cross-encoder vs bi-encoder architectures for reranking"
```

**Domain-Specific Queries:**
```
"Techniques for low-resource neural machine translation"
"Privacy-preserving machine learning methods"
"Graph neural networks for molecular property prediction"
```

## 🔬 Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Claude API (Sonnet 4) | Analysis, synthesis, reasoning |
| **Orchestration** | LangGraph | Agentic workflow management |
| **Embeddings** | SPECTER2 | Scientific paper embeddings |
| **Lexical Retrieval** | BM25 (rank-bm25) | Keyword-based search |
| **Re-ranking** | Cross-Encoder | Precision optimization |
| **UI** | Streamlit | Web interface |
| **Data Sources** | arXiv, Semantic Scholar, Google Scholar | Paper retrieval |

## 📈 Performance Considerations

**Optimization Tips:**

1. **Caching**: Implement caching for repeated queries
2. **Batch Processing**: Process embeddings in batches
3. **Lazy Loading**: Load models on-demand
4. **Rate Limiting**: Respect API rate limits
5. **Parallel Retrieval**: Query multiple sources concurrently

**Typical Response Times:**
- Search (50 papers): 2-5 seconds
- Embedding (50 papers): 3-10 seconds
- Re-ranking (20 papers): 1-3 seconds
- LLM Analysis: 5-15 seconds
- **Total**: ~15-30 seconds per query

## 🐛 Troubleshooting

**Common Issues:**

1. **"ANTHROPIC_API_KEY is required"**
   - Ensure `.env` file exists with valid API key
   - Check that `.env` is in the project root directory

2. **Slow embedding generation**
   - First run downloads SPECTER2 model (~500MB)
   - Subsequent runs use cached model
   - Use GPU if available for faster inference

3. **Rate limit errors**
   - Semantic Scholar: 100 requests per 5 minutes (free tier)
   - Add API key for higher limits
   - Implement exponential backoff

4. **Out of memory**
   - Reduce `TOP_K_RETRIEVAL` and `TOP_K_RERANK`
   - Process papers in smaller batches
   - Use `faiss-cpu` instead of full embeddings in memory

5. **Cross-encoder not available**
   - System falls back to bi-encoder similarity
   - Install sentence-transformers with cross-encoder support:
     ```bash
     pip install sentence-transformers[cross-encoder]
     ```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{alexender2024research,
  title={Research Assistant AI: Hybrid Retrieval and Agentic Workflows for Literature Discovery},
  author={Alexender, Manoj and Aldwake, Seraj},
  year={2024},
  note={ECE 579 Intelligent Systems Project}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Anthropic](https://www.anthropic.com/) for Claude API
- [Allen AI](https://allenai.org/) for Semantic Scholar and SPECTER2
- [arXiv](https://arxiv.org/) for open access to scientific papers
- [LangChain](https://www.langchain.com/) for LangGraph framework

## 📮 Contact

**Manoj Alexender** - [Your Email]  
**Seraj Aldwake** - [Your Email]

**Project Link:** [Your GitHub Repository]

---

Built with ❤️ for ECE 579 Intelligent Systems
