# Installation & Deployment Guide

Complete guide for setting up and deploying the Research Assistant AI system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Testing](#testing)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)
7. [Production Deployment](#production-deployment)

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Disk**: 2GB free space (for models and cache)
- **Internet**: Required for API calls

### Recommended Specifications
- **Python**: 3.10+
- **RAM**: 8GB+
- **CPU**: 4+ cores
- **GPU**: Optional (speeds up embedding generation)

## Installation

### Step 1: Clone/Download the Project

```bash
# If using git
git clone <repository-url>
cd research-assistant-ai

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install approximately 20 packages including:
- `anthropic` - Claude API client
- `langchain` & `langgraph` - Agent orchestration
- `sentence-transformers` - Embeddings
- `streamlit` - Web interface
- `arxiv` - arXiv API client
- And more...

**Installation time**: 2-5 minutes depending on internet speed.

### Step 4: Download Required Models

On first run, the system will automatically download:
- **SPECTER2** embeddings model (~500MB)
- **Cross-encoder** for re-ranking (~100MB)

This happens automatically but may take 5-10 minutes.

## Configuration

### Environment Variables

1. **Copy the template:**
```bash
cp .env.template .env
```

2. **Edit `.env` file:**
```env
# Required: Get from https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# Optional: Get from https://serpapi.com/
SERPAPI_KEY=your_serpapi_key_here

# Optional: For higher rate limits
SEMANTIC_SCHOLAR_API_KEY=

# Configuration
MAX_RESULTS=10
EMBEDDING_MODEL=sentence-transformers/allenai-specter2
LLM_MODEL=claude-sonnet-4-20250514
```

### Getting API Keys

#### Anthropic API Key (Required)
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new key
5. Copy and paste into `.env`

**Pricing**: Pay-as-you-go (starts at $3 per million input tokens)

#### SerpAPI Key (Optional)
1. Go to https://serpapi.com/
2. Sign up for free account
3. Copy your API key
4. Paste into `.env`

**Free tier**: 100 searches/month

### Advanced Configuration

Edit `config.py` to customize:

```python
# Retrieval settings
TOP_K_RETRIEVAL = 20      # Papers to retrieve
TOP_K_RERANK = 10         # Papers after re-ranking

# arXiv categories to search
ARXIV_CATEGORIES = [
    "cs.AI",      # Artificial Intelligence
    "cs.CL",      # Computation and Language
    "cs.CV",      # Computer Vision
    "cs.LG",      # Machine Learning
    "cs.IR",      # Information Retrieval
    "stat.ML",    # Statistics - Machine Learning
]

# LLM settings
MAX_TOKENS = 4000
TEMPERATURE = 0.3
```

## Testing

### Quick System Test

Run the comprehensive test suite:

```bash
python test_system.py
```

**What it tests:**
- ✅ Package imports
- ✅ Configuration validity
- ✅ API connectivity
- ✅ Data retrieval
- ✅ Embedding model
- ✅ Claude API
- ✅ Hybrid retrieval
- ✅ End-to-end workflow

**Expected output:**
```
======================================================================
RESEARCH ASSISTANT AI - SYSTEM TEST
======================================================================

Testing imports...
  ✓ Anthropic
  ✓ LangChain
  ...
✅ All required packages installed

Testing configuration...
  ✓ .env file found
  ✓ ANTHROPIC_API_KEY configured
✅ Configuration valid

...

======================================================================
TEST SUMMARY
======================================================================
✅ PASS - Package Imports
✅ PASS - Configuration
✅ PASS - Data Retrieval
✅ PASS - Embedding Model
✅ PASS - Claude API
✅ PASS - Hybrid Retrieval
✅ PASS - Agent Workflow

Results: 7/7 tests passed

🎉 All tests passed! Your system is ready to use.
======================================================================
```

### Manual Testing

Test each component individually:

```bash
# Test CLI
python cli.py "transformer architecture" --verbose

# Test web interface
streamlit run app.py

# Test Python API
python examples.py
```

## Usage

### Option 1: Web Interface (Recommended)

**Start the server:**
```bash
streamlit run app.py
```

**Access:** Open browser to http://localhost:8501

**Features:**
- Interactive search box
- Parameter adjustment (BM25 weight, top-k)
- Data source selection
- Search history
- Detailed paper views
- Export results

### Option 2: Command Line

**Basic query:**
```bash
python cli.py "retrieval augmented generation"
```

**With context:**
```bash
python cli.py "few-shot learning" --context "low-resource NLP"
```

**Verbose output:**
```bash
python cli.py "graph neural networks" --verbose
```

### Option 3: Python API

Create a script `my_research.py`:

```python
from agent import research_query

# Simple query
result = research_query("neural machine translation")

# Print synthesis
print(result['synthesis'])

# Access papers
for i, paper in enumerate(result['reranked_papers'][:5], 1):
    print(f"{i}. {paper['title']}")
    print(f"   Authors: {', '.join(paper['authors'][:3])}")
    print()
```

Run it:
```bash
python my_research.py
```

## Troubleshooting

### Issue 1: "ANTHROPIC_API_KEY is required"

**Solution:**
1. Ensure `.env` file exists in project root
2. Check that `ANTHROPIC_API_KEY=` line has your key
3. Restart terminal/IDE after editing `.env`
4. Verify key is valid at https://console.anthropic.com/

### Issue 2: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'anthropic'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 3: Slow First Run

**Symptom:** First query takes 2-3 minutes

**Explanation:** System is downloading SPECTER2 model (~500MB)

**Solution:** This is normal. Subsequent runs are much faster (15-30s).

### Issue 4: Rate Limit Errors

**Error:**
```
Error 429: Rate limit exceeded
```

**Solution:**
- Wait a few minutes between queries
- For Semantic Scholar: Add API key for higher limits
- Consider caching results for repeated queries

### Issue 5: Out of Memory

**Error:**
```
OutOfMemoryError or system freeze
```

**Solution:**
Edit `config.py`:
```python
TOP_K_RETRIEVAL = 10    # Reduce from 20
TOP_K_RERANK = 5        # Reduce from 10
MAX_RESULTS = 5         # Reduce from 10
```

### Issue 6: Streamlit Port Already in Use

**Error:**
```
Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9  # macOS/Linux
```

### Issue 7: SSL Certificate Errors

**Error:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution (macOS):**
```bash
/Applications/Python*/Install\ Certificates.command
```

**Solution (Windows/Linux):**
```bash
pip install --upgrade certifi
```

## Production Deployment

### Option A: Docker Deployment

**1. Create Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**2. Build and run:**

```bash
# Build image
docker build -t research-assistant .

# Run container
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=your_key_here \
  research-assistant
```

### Option B: Cloud Deployment (AWS)

**1. Set up EC2 instance:**
```bash
# Launch Ubuntu 22.04 instance (t3.medium recommended)
# SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# Install dependencies
sudo apt update
sudo apt install python3.10 python3-pip git

# Clone repository
git clone <repo-url>
cd research-assistant-ai

# Install packages
pip3 install -r requirements.txt
```

**2. Configure environment:**
```bash
# Create .env
nano .env
# Add your API keys

# Test installation
python3 test_system.py
```

**3. Run with systemd:**
```bash
# Create service file
sudo nano /etc/systemd/system/research-assistant.service
```

Add:
```ini
[Unit]
Description=Research Assistant AI
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/research-assistant-ai
Environment="PATH=/home/ubuntu/.local/bin"
ExecStart=/usr/bin/python3 -m streamlit run app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

**4. Start service:**
```bash
sudo systemctl enable research-assistant
sudo systemctl start research-assistant
sudo systemctl status research-assistant
```

**5. Configure nginx (optional):**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Option C: Streamlit Cloud

**1. Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

**2. Deploy:**
1. Go to https://streamlit.io/cloud
2. Connect GitHub account
3. Select your repository
4. Add secrets (ANTHROPIC_API_KEY, etc.)
5. Deploy!

### Performance Optimization

**1. Enable caching:**

Add to `agent.py`:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str):
    # Your search logic
    pass
```

**2. Use Redis for distributed caching:**
```bash
pip install redis

# In code
import redis
r = redis.Redis(host='localhost', port=6379)
```

**3. Batch processing:**
```python
# Process multiple queries efficiently
queries = ["query1", "query2", "query3"]
results = [research_query(q) for q in queries]
```

**4. GPU acceleration:**
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Models will automatically use GPU if available
```

### Monitoring

**1. Add logging:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

**2. Track metrics:**
```python
import time

def track_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logging.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper
```

### Security Best Practices

1. **Never commit API keys** - use `.gitignore`
2. **Use environment variables** - never hardcode secrets
3. **Implement rate limiting** - prevent abuse
4. **Add authentication** - for production use
5. **Use HTTPS** - encrypt traffic
6. **Sanitize inputs** - prevent injection attacks

## Support & Resources

- **Documentation**: See `README.md` for full details
- **Examples**: Check `examples.py` for usage patterns
- **Testing**: Run `test_system.py` to verify setup
- **Quick Start**: See `QUICKSTART.md` for fast setup

## Next Steps

After successful installation:

1. ✅ Run test suite: `python test_system.py`
2. ✅ Try example queries: `python examples.py`
3. ✅ Launch web interface: `streamlit run app.py`
4. ✅ Read documentation: Check `README.md`
5. ✅ Customize configuration: Edit `config.py`

Happy researching! 🔬
