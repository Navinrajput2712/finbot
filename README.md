# 💰 FinBot — AI-Powered Financial Advisory Chatbot

> An end-to-end GenAI project combining RAG pipeline + LLaMA 3.1 8B via NVIDIA NIM API + FastAPI + Streamlit

---

## 🚀 Live Demo
> Deploy on Hugging Face Spaces and add link here

---

## 📌 Project Overview

FinBot is a production-ready AI financial advisory chatbot built for Indian users. It combines a Retrieval-Augmented Generation (RAG) pipeline with LLaMA 3.1 8B (via NVIDIA NIM API) to answer questions about personal finance with grounded, citation-backed responses.

Unlike generic chatbots, FinBot retrieves relevant content from a curated financial knowledge base (RBI, SEBI, Income Tax, AMFI, IRDAI documents) before generating answers — ensuring accuracy and minimizing hallucinations.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI (port 8501)
    │  POST /chat
    ▼
FastAPI Backend (port 8000)
    │
    ├─► Query Router
    │       ├── Market query? → yfinance API → Live price
    │       └── Knowledge query → RAG Pipeline
    │
    ▼
RAG Pipeline
    │
    ├─► ChromaDB (semantic search, top-6 chunks)
    │       └── BAAI/bge-base-en-v1.5 embeddings
    │
    ├─► Cross-Encoder Reranker (ms-marco-MiniLM)
    │       └── Rerank to top-4 most relevant chunks
    │
    └─► NVIDIA NIM API
            └── meta/llama-3.1-8b-instruct
                    └── Grounded response + disclaimer
                            │
                            ▼
                    Answer + Sources + Confidence Score
```

---

## ⚡ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | LLaMA 3.1 8B Instruct | Answer generation |
| LLM API | NVIDIA NIM API | Hosted model inference |
| RAG | LangChain + ChromaDB | Document retrieval |
| Embeddings | BAAI/bge-base-en-v1.5 | Semantic search |
| Reranker | ms-marco-MiniLM-L-6-v2 | Result reranking |
| Backend | FastAPI + Uvicorn | REST API |
| Frontend | Streamlit | Chat UI |
| Market Data | yfinance | Live stock prices |
| DevOps | Docker + docker-compose | Containerisation |

---

## 📊 Evaluation Results

| Domain | Avg Confidence | Avg Latency | Answered |
|--------|---------------|-------------|----------|
| Budgeting | 0.85 | ~2s | 4/4 |
| Investing | 0.83 | ~2s | 4/4 |
| Taxation | 0.89 | ~2s | 4/4 |
| Insurance | 0.81 | ~2s | 4/4 |
| Loans | 0.84 | ~2s | 4/4 |
| **OVERALL** | **0.84** | **~2s** | **20/20** |

> Hallucination rate: < 5% (RAG-grounded responses only)
> All responses include source citations and AI disclaimer

---

## 🗂️ Project Structure

```
finbot/
├── rag/
│   ├── ingest.py          # PDF → chunks → ChromaDB
│   ├── retriever.py       # ChromaDB + cross-encoder reranker
│   ├── pipeline.py        # Full RAG chain with NVIDIA NIM
│   └── test_pipeline.py   # 10-query domain test
├── backend/
│   ├── main.py            # FastAPI app entry point
│   ├── schemas.py         # Pydantic v2 models
│   ├── llm_loader.py      # NVIDIA NIM client
│   ├── market_data.py     # yfinance live data
│   └── routes/
│       ├── chat.py        # POST /chat endpoint
│       └── health.py      # GET /health endpoint
├── frontend/
│   └── app.py             # Streamlit chat UI
├── data/
│   └── knowledge_base/    # Financial PDFs (RBI, SEBI, ITD etc.)
├── evaluation/
│   └── evaluate.py        # 20-query evaluation script
├── chroma_db/             # Persisted vector store
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
├── Dockerfile.backend     # Backend container
├── Dockerfile.frontend    # Frontend container
└── docker-compose.yml     # Multi-service orchestration
```

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.11+
- NVIDIA NIM API key (free at https://build.nvidia.com)

### Step 1 — Clone and setup
```bash
git clone https://github.com/yourusername/finbot.git
cd finbot
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Step 2 — Configure environment
```bash
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY
```

### Step 3 — Add PDFs to knowledge base
```
Place financial PDFs in: data/knowledge_base/
Recommended: RBI, SEBI, Income Tax, AMFI, IRDAI guides
```

### Step 4 — Run RAG ingestion
```bash
python -m rag.ingest
```

### Step 5 — Start backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 6 — Start frontend (new terminal)
```bash
streamlit run frontend/app.py
```

Open: http://localhost:8501

### Docker (optional)
```bash
docker-compose up --build
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| NVIDIA_API_KEY | ✅ Yes | NVIDIA NIM API key |
| NVIDIA_BASE_URL | ✅ Yes | https://integrate.api.nvidia.com/v1 |
| NVIDIA_MODEL | ✅ Yes | meta/llama-3.1-8b-instruct |
| CHROMA_DB_PATH | ✅ Yes | ./chroma_db |
| KNOWLEDGE_BASE_PATH | ✅ Yes | ./data/knowledge_base |
| ALPHA_VANTAGE_KEY | ❌ Optional | Extra market data |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API status |
| POST | /chat | Main chat endpoint |
| GET | /health | System health check |
| GET | /market/{ticker} | Live stock data |
| POST | /ingest | Upload PDF to knowledge base |
| GET | /docs | Swagger UI |

*Built with ❤️ using NVIDIA NIM + LLaMA 3.1 8B + LangChain + ChromaDB*
