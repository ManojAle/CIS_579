# 🔬 Research Assistant AI

> **An intelligent research assistant powered by hybrid retrieval, agentic workflows, and Claude AI for academic literature discovery and synthesis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

**Authors:** Manoj Alexender & Seraj Aldwake  
**Course:** ECE 579 Intelligent Systems  
**Institution:** University of Michigan-Dearborn

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

Research Assistant AI is an advanced literature discovery system that combines state-of-the-art retrieval techniques with AI-powered analysis. It helps researchers:

- **Discover** relevant academic papers across multiple sources (arXiv, Semantic Scholar, Google Scholar)
- **Retrieve** papers using hybrid search (BM25 + semantic embeddings)
- **Rank** results with cross-encoder re-ranking for maximum precision
- **Analyze** and synthesize findings using Claude AI
- **Compare** different approaches and identify research gaps

Built for ECE 579 Intelligent Systems, this project demonstrates practical applications of information retrieval, natural language processing, and agentic AI workflows.

---

## ✨ Key Features

### 🔍 **Multi-Source Search**
- Integration with arXiv, Semantic Scholar, and Google Scholar APIs
- Intelligent query routing and result deduplication
- Parallel retrieval for faster results

### 🧠 **Hybrid Retrieval System**
- **BM25 (Lexical)**: Traditional keyword-based search for precise term matching
- **SPECTER2 (Semantic)**: Scientific paper embeddings for conceptual similarity
- **Weighted Fusion**: Configurable combination of both approaches

### 📊 **Intelligent Re-ranking**
- Cross-encoder models for fine-grained relevance scoring
- Configurable top-k filtering
- Fallback to bi-encoder when needed

### 🤖 **Agentic Workflows**
- LangGraph orchestration for multi-step research tasks
- Automatic error handling and retry logic
- State management for complex queries

### 💡 **AI-Powered Analysis**
- Claude API integration for synthesis and insights
- Grounded responses with citation tracking
- Comparative analysis and research gap identification

### 🎨 **Interactive Interface**
- Streamlit-based web UI
- Real-time search and analysis
- Search history and session management
- Adjustable retrieval parameters

### 📈 **Comprehensive Evaluation**
- IR metrics: Recall@k, Precision@k, nDCG, MRR
- RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision/Recall
- BEIR benchmark support

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Query                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Source Retrieval                        │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────┐      │
│  │  arXiv   │    │ Semantic Scholar │    │    Scholar   │      │
│  └──────────┘    └──────────────────┘    └──────────────┘      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Hybrid Retrieval Engine                      │
│  ┌─────────────────┐              ┌──────────────────────┐      │
│  │  BM25 (Lexical) │              │  SPECTER2 (Semantic) │      │
│  │  - Tokenization │              │  - Embeddings        │      │
│  │  - Term weights │              │  - Cosine similarity │      │
│  └────────┬────────┘              └──────────┬───────────┘      │
│           │                                   │                  │
│           └───────────┬───────────────────────┘                  │
│                       │                                          │
│              ┌────────▼────────┐                                 │
│              │  Score Fusion   │                                 │
│              │  (α*BM25 + β*S2)│                                 │
│              └────────┬────────┘                                 │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cross-Encoder Re-ranking                      │
│  - Fine-grained relevance scoring                                │
│  - Top-k selection                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Claude AI Analysis                          │
│  - Synthesis and insights                                        │
│  - Comparative analysis                                          │
│  - Citation generation                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Results & Citations                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- SerpAPI key (optional, for Google Scholar)

### Installation

```bash
# Clone the repository
git clone https://github.com/ManojAle/CIS_579.git
cd CIS_579

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.template .env
# Edit .env and add your API keys
```

### Run the Web Interface

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## 📦 Installation

### Detailed Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ManojAle/CIS_579.git
   cd CIS_579
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the project root:
   ```env
   # Required
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Optional
   SERPAPI_KEY=your_serpapi_key_here
   SEMANTIC_SCHOLAR_API_KEY=your_s2_api_key_here
   
   # Configuration
   MAX_RESULTS=10
   EMBEDDING_MODEL=sentence-transformers/allenai-specter2
   LLM_MODEL=claude-sonnet-4-20250514
   TOP_K_RETRIEVAL=20
   TOP_K_RERANK=10
   TEMPERATURE=0.3
   ```

### API Key Setup

- **Anthropic API** (Required): https://console.anthropic.com/
- **SerpAPI** (Optional): https://serpapi.com/
- **Semantic Scholar** (Optional): https://www.semanticscholar.org/product/api

---

## 💻 Usage

### Option 1: Web Interface (Recommended)

```bash
streamlit run app.py
```

**Features:**
- Interactive search interface
- Adjustable retrieval parameters
- Data source selection
- Search history
- Detailed paper views
- Export results

### Option 2: Command Line Interface

**Basic search:**
```bash
python cli.py "retrieval augmented generation in conversational AI"
```

**With context:**
```bash
python cli.py "transformer attention mechanisms" --context "I'm building a chatbot for healthcare"
```

**Verbose mode:**
```bash
python cli.py "few-shot learning" --verbose
```

### Option 3: Python API

**Simple usage:**
```python
from agent import research_query

# Basic query
result = research_query("neural machine translation")

# Access results
print(result['synthesis'])
print(f"Found {len(result['reranked_papers'])} papers")
for paper in result['reranked_papers'][:5]:
    print(f"- {paper['title']}")
```

**Advanced usage:**
```python
from agent import ResearchAgent
from config import Config

# Initialize agent
agent = ResearchAgent()

# Run with full control
result = agent.run(
    query="large language model reasoning",
    user_context="Focus on chain-of-thought prompting",
    max_results=15,
    bm25_weight=0.6
)

# Access detailed results
print(f"Papers found: {len(result['raw_papers'])}")
print(f"Retrieved: {len(result['retrieved_papers'])}")
print(f"Final: {len(result['reranked_papers'])}")
print(f"Analysis:\n{result['synthesis']}")
print(f"\nCitations:\n{result['citations']}")
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | - | **Required** - Claude API key |
| `SERPAPI_KEY` | - | Optional - Google Scholar access |
| `SEMANTIC_SCHOLAR_API_KEY` | - | Optional - Higher rate limits |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `EMBEDDING_MODEL` | `allenai-specter2` | Embedding model |
| `MAX_RESULTS` | `10` | Maximum papers to return |
| `TOP_K_RETRIEVAL` | `20` | Papers before re-ranking |
| `TOP_K_RERANK` | `10` | Papers after re-ranking |
| `TEMPERATURE` | `0.3` | LLM temperature (0.0-1.0) |
| `BM25_WEIGHT` | `0.5` | Weight for BM25 (0.0-1.0) |

### Configuration in Code

```python
from config import Config

# Custom configuration
config = Config(
    max_results=15,
    top_k_retrieval=30,
    top_k_rerank=15,
    bm25_weight=0.6,  # More weight to lexical search
    temperature=0.2    # More deterministic outputs
)
```

---

## 📊 Evaluation

### Running Evaluations

```python
from evaluation import IRMetrics, RAGASMetrics, print_evaluation_report

# Example retrieval evaluation
retrieved_ids = ['paper1', 'paper2', 'paper3', 'paper4', 'paper5']
relevant_ids = ['paper2', 'paper4', 'paper6']
relevance_scores = {
    'paper1': 0, 'paper2': 3, 'paper3': 0,
    'paper4': 2, 'paper5': 0, 'paper6': 3
}

metrics = IRMetrics.evaluate_retrieval(
    retrieved_ids,
    relevant_ids,
    relevance_scores,
    k_values=[5, 10]
)

print_evaluation_report(metrics)
```

### Evaluation Metrics

**Information Retrieval Metrics:**
- **Recall@k**: Proportion of relevant documents retrieved
- **Precision@k**: Proportion of retrieved documents that are relevant
- **nDCG@k**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

**RAG-Specific Metrics:**
- **Faithfulness**: Grounding in retrieved context
- **Answer Relevancy**: Relevance to the query
- **Context Precision**: Relevance of retrieved context
- **Context Recall**: Coverage of relevant information

---

## 📁 Project Structure

```
research-assistant-ai/
│
├── agent.py                 # LangGraph workflow & agentic orchestration
├── app.py                   # Streamlit web interface
├── app_enhanced.py          # Enhanced UI with additional features
├── cli.py                   # Command-line interface
├── config.py                # Configuration management
├── evaluation.py            # Evaluation metrics (BEIR, RAGAS)
├── hybrid_retrieval.py      # Hybrid retrieval & re-ranking engine
├── retrieval.py             # Multi-source data retrieval
├── enhanced_agent.py        # Extended agent capabilities
├── examples.py              # Usage examples
│
├── requirements.txt         # Python dependencies
├── .env.template            # Environment variable template
├── .env                     # Your API keys (gitignored)
│
├── README.md               # This file
├── PROJECT_SUMMARY.md      # Project overview
└── INSTALLATION.md         # Detailed installation guide
```

---

## 🛠️ Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Claude API (Sonnet 4) | Analysis, synthesis, reasoning |
| **Orchestration** | LangGraph | Agentic workflow management |
| **Embeddings** | SPECTER2 | Scientific paper embeddings |
| **Lexical Search** | BM25 (rank-bm25) | Keyword-based retrieval |
| **Re-ranking** | Cross-Encoder | Precision optimization |
| **UI** | Streamlit | Web interface |
| **Data Sources** | arXiv, Semantic Scholar, Google Scholar | Paper retrieval |
| **Evaluation** | BEIR, RAGAS | Quality assessment |

---

## 📈 Performance

### Typical Response Times

| Operation | Time Range | Notes |
|-----------|------------|-------|
| Multi-source search (50 papers) | 2-5 sec | Parallel API calls |
| Embedding generation (50 papers) | 3-10 sec | First run downloads model |
| Re-ranking (20 papers) | 1-3 sec | Cross-encoder inference |
| Claude analysis | 5-15 sec | Depends on context length |
| **Total end-to-end** | **15-30 sec** | Per query |

### Optimization Tips

1. **Enable Caching**: Cache embeddings and frequent queries
2. **Batch Processing**: Process papers in batches for embeddings
3. **Lazy Loading**: Load models on-demand
4. **Parallel Retrieval**: Query multiple sources concurrently
5. **GPU Acceleration**: Use CUDA for faster embedding generation

### Resource Requirements

- **RAM**: 4-8 GB minimum
- **Disk**: 2 GB (for model cache)
- **GPU**: Optional but recommended for faster embeddings

---

## 🧪 Example Queries

### Broad Research Questions
```python
"What are the latest advances in multimodal learning?"
"Compare different approaches to few-shot learning"
"Research gaps in explainable AI"
```

### Specific Technical Questions
```python
"How does SPECTER2 improve on SPECTER for paper embeddings?"
"Evaluation metrics for retrieval augmented generation systems"
"Cross-encoder vs bi-encoder architectures for reranking"
```

### Domain-Specific Queries
```python
"Techniques for low-resource neural machine translation"
"Privacy-preserving machine learning methods"
"Graph neural networks for molecular property prediction"
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "ANTHROPIC_API_KEY is required"**
- Ensure `.env` file exists in project root
- Verify API key is valid and not expired
- Check for extra spaces or quotes in `.env`

**2. Slow Embedding Generation**
- First run downloads SPECTER2 model (~500MB)
- Subsequent runs use cached model
- Use GPU for 5-10x speedup
- Reduce `TOP_K_RETRIEVAL` for faster processing

**3. Rate Limit Errors**
- Semantic Scholar: 100 requests per 5 minutes (free tier)
- Add API key for higher limits
- Implement exponential backoff (built-in)
- Reduce `MAX_RESULTS` parameter

**4. Out of Memory**
- Reduce `TOP_K_RETRIEVAL` and `TOP_K_RERANK`
- Process papers in smaller batches
- Use `faiss-cpu` for efficient vector storage
- Close other applications to free RAM

**5. Cross-Encoder Not Available**
- System automatically falls back to bi-encoder
- Install with: `pip install sentence-transformers`
- Requires ~2GB disk space for model

**6. "Module not found" Errors**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# Or install specific missing package
pip install <package_name>
```

**7. Streamlit Won't Start**
```bash
# Check if port 8501 is in use
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Use different port
streamlit run app.py --server.port 8502
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use CLI verbose flag:
```bash
python cli.py "your query" --verbose
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork the repository
git clone https://github.com/YOUR_USERNAME/CIS_579.git
cd CIS_579

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git add .
git commit -m 'Add amazing feature'

# Push to your fork
git push origin feature/amazing-feature

# Open a Pull Request
```

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Include examples for new functionality
- Test thoroughly before submitting PR

### Areas for Contribution

- Additional data source integrations
- New evaluation metrics
- Performance optimizations
- UI/UX improvements
- Bug fixes and error handling
- Documentation improvements

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{alexender2024research,
  title={Research Assistant AI: Hybrid Retrieval and Agentic Workflows for Literature Discovery},
  author={Alexender, Manoj and Aldwake, Seraj},
  year={2024},
  institution={University of Michigan-Dearborn},
  note={ECE 579 Intelligent Systems Project}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Anthropic** for Claude API and excellent documentation
- **Allen AI** for Semantic Scholar API and SPECTER2 embeddings
- **arXiv** for open access to scientific literature
- **LangChain** for LangGraph framework
- **Streamlit** for the intuitive web framework
- **Professor Yi Lu Murphey** for guidance in ECE 579

---

## 📮 Contact

**Manoj Alexender** - GitHub: [@ManojAle](https://github.com/ManojAle)  
**Seraj Aldwake** - Collaborator

**Project Repository:** https://github.com/ManojAle/CIS_579

---

## 🗺️ Roadmap

### Current Version (v1.0)
- ✅ Multi-source retrieval (arXiv, Semantic Scholar, Google Scholar)
- ✅ Hybrid retrieval (BM25 + SPECTER2)
- ✅ Cross-encoder re-ranking
- ✅ Claude AI integration
- ✅ Streamlit web interface
- ✅ Evaluation metrics (BEIR, RAGAS)

### Planned Features (v2.0)
- [ ] User authentication and personalization
- [ ] Save and export research reports
- [ ] Citation graph visualization
- [ ] Batch query processing
- [ ] API endpoint for integration
- [ ] Docker containerization
- [ ] Advanced filtering options
- [ ] Collaborative research features

---

<div align="center">

**Built with ❤️ for ECE 579 Intelligent Systems**

[⬆ Back to Top](#-research-assistant-ai)

</div>
