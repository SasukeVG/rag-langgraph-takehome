# AI Engineer Take-Home — RAG Application (LangChain + LangGraph)

## Overview

This project is a small, outcome-focused Retrieval-Augmented Generation (RAG) application built as part of an AI Engineer take-home assignment.

The goal is to demonstrate:

- **Grounded question answering** over a tiny local Markdown dataset
- **Multi-step orchestration** with decision-making (answer vs clarification)
- **Short-term conversational memory**
- **Incremental feedback** during execution
- **Basic reliability and logging**
- **A clear, easy-to-run CLI and API interface**


---

## Key Features

- **Local RAG pipeline**
  - Loads a small set of Markdown documents from the local `data/` directory
  - Chunks, embeds, and retrieves relevant context using FAISS

- **Multi-step orchestration**
  - Implemented with LangGraph (`RAGGraph` in `app/graph.py`)
  - Includes retrieval, decision, and conditional branching (answer vs clarification)

- **Short-term memory**
  - Conversation state is stored in a simple in-memory buffer (`SessionMemory` in `app/memory.py`)
  - Used by the CLI to preserve context across multiple turns in a single session

- **Incremental feedback**
  - **CLI**: token-level streaming output from the LLM
  - **API**: step-based execution feedback (retrieve → decision → answer/clarify) with per-step status

- **Reliability**
  - Input validation via Pydantic models
  - Retry with backoff for LLM calls in the LangGraph nodes
  - Clear, human-readable structured logging via Loguru

- **Interfaces**
  - **CLI** (primary, most explicit streaming/memory behavior)
  - **FastAPI backend** (optional, mirrors the same RAG core)

---

## Tech Stack

- **Python**: 3.12+
- **LangChain**: v0.3.x
- **LangGraph**: v0.6.x
- **Embeddings**: `intfloat/e5-base-v2` (English-only, retrieval-optimized)
- **Vector Store**: FAISS (local, in-memory)
- **LLM Provider**: OpenRouter-compatible chat models via `langchain-openai`
- **API**: FastAPI + Uvicorn
- **Config & settings**: `pydantic-settings` + `.env`
- **Logging**: Loguru

---


## Setup

### 1. Prerequisites

- **Python** 3.12 or newer
- A virtual environment (recommended)
- An **OpenRouter API key** for an OpenRouter-compatible LLM

The project assumes an environment where outbound HTTPS is allowed from the machine running the code.

---

### 2. Install dependencies

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

This uses `pyproject.toml` as the single source of truth for dependencies and installs the project in editable mode.

If you prefer `uv`:

```bash
uv sync
```

---

### 3. Environment configuration

From the repository root, create your `.env`:

```bash
cp example.env .env
```

Then edit `.env` and at minimum set:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

Other relevant variables (with sensible defaults defined in `config.py` and `example.env`):

- **OpenRouter**
  - `OPENROUTER_MODEL` — model name (e.g. `mistralai/devstral-2512:free`)
  - `OPENROUTER_BASE_URL` — base URL for the OpenRouter API
- **Retrieval**
  - `RETRIEVAL_DATA_DIR` — path to the Markdown corpus (default: `data`)
  - `RETRIEVAL_DISTANCE_THRESHOLD` — threshold for deciding clarification vs answering
- **API**
  - `API_HOST`, `API_PORT`, `API_RELOAD_SERVER`
- **Logging**
  - `LOG_LEVEL` — e.g. `DEBUG`, `INFO`

Environment variables are loaded from `.env` by `config.py` on startup.

---

## Running the Application

### Option A — CLI (Primary Interface)

The CLI is the reference implementation and demonstrates **streaming output** and **memory** most clearly.

From the repository root:

```bash
PYTHONPATH=src python src/app/main.py
```

Example interaction:

```text
======================================================================
AI Engineer Take-Home: RAG Application
======================================================================

Initializing...
Ready! Type 'help' for commands or 'quit' to exit.

Commands:
  ask <question>  - Ask a question
  help           - Show this help message
  quit / exit    - Exit the application

> ask What are the key features of this RAG system?
[INFO] Retrieved 3 documents:
  1. doc2.md (distance: 0.274)
  ...

[assistant] ... streamed answer tokens ...
```

Behavior highlights:

- **Responses** are streamed token-by-token to stdout.
- **Retrieval** details (documents and distances) are printed for transparency.
- **Conversation memory** is preserved across turns for the duration of the CLI session.

### Example Queries (CLI Benchmarks)

```text
> ask whats the main benefits of using Retrieval-Augmented Generation?

> ask tell me difference between faiss and chroma vector bases

> ask what key features of langgraph

> ask describe the main steps of the RAG pipeline

> ask when should you choose faiss over chroma for a rag application
```

---

### Option B — FastAPI Backend (Optional)

The API mirrors the CLI behavior and uses the same RAG core (`RAGGraph` + `DocumentRetriever`).

From the repository root:

```bash
python run_api.py
```

`run_api.py`:

- Adds `src/` to `sys.path` so imports like `config` and `app.graph` work from the repository root.
- Configures logging using `logging_config.setup_logging`.
- Starts Uvicorn with host/port/reload read from `API_*` settings in `.env`.

By default (see `config.APISettings`) the server listens on:

- `http://0.0.0.0:8008`

---

## API Endpoints

### `GET /health`

Simple health check.

```json
{ "status": "healthy", "service": "rag-document-api" }
```

### `GET /stats`

Returns basic information about the vector store and retrieval configuration.

Response (simplified):

```json
{
  "status": "ready",
  "documents": 5,
  "total_chunks": 120,
  "embedding_model": "intfloat/e5-base-v2",
  "distance_threshold": 0.9,
  "context_top_k": 2
}
```

### `POST /ask`

Primary endpoint to run the RAG pipeline.

Request:

```json
{
  "query": "What are the key features of this RAG system?"
}
```

Response (simplified):

```json
{
  "status": "completed",
  "steps": [
    { "step": "retrieve", "status": "done", "docs_found": 3 },
    { "step": "decision", "status": "done", "clarify": false },
    { "step": "answer", "status": "done" }
  ],
  "answer": "...",
  "documents": [
    { "source": "doc2.md", "score": 0.2743 }
  ]
}
```

Notes:

- `steps` reflects the internal LangGraph workflow:
  - `retrieve` → `decision` → `answer` **or** `clarify`
- When clarification is required (`clarify: true`), the third step is:
  - `{ "step": "clarify", "status": "done" }`

---

### Swagger UI

Swagger UI is available at:

- `http://localhost:<API_PORT>/docs` (e.g. `http://localhost:8008/docs`)

Notes:

- Swagger UI does **not** support token streaming.
- Incremental feedback is exposed instead through explicit **step statuses** in the `steps` field of `AskResponse`.

---

## Next Steps

Planned improvements and extensions are documented in [`NEXT_STEPS.md`](./NEXT_STEPS.md).

---


It is designed to be easy to run, easy to review, and easy to extend.