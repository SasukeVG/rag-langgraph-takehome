# Next Steps

- **Reranking of retrieved context**
  - Add a lightweight reranking step (e.g. cross-encoder) on top of FAISS results to improve precision when multiple chunks are loosely relevant.

- **More granular thresholds**
  - Separate thresholds for:
    - deciding whether to answer vs. ask for clarification, and
    - selecting which chunks are included in the final context.
  - Expose these values via configuration.

- **Improved structured outputs**
  - Further optimize prompts for list-based questions (benefits, steps, features).
  - Where appropriate, return both a natural-language answer and a structured representation.

- **Memory handling**
  - Migrate memory management to newer LangChain abstractions as they stabilize.
  - Optionally add lightweight persistence for API sessions.

- **Centralized RAG configuration and model selection**
  - Decouple RAG configuration (LLM model, embedding model, thresholds, top-k, prompts) from environment variables and hardcoded values.
  - Enable safe experimentation and quick switching between models and retrieval strategies without code changes.

- **Asynchronous API execution**
  - Replace the current request/response execution model with an asynchronous job-based workflow.
  - Allow multiple concurrent queries with polling or subscription-based result delivery.

- **External vector store deployment**
  - Decouple vector storage from the application process.
  - Support a shared or persistent vector store to enable horizontal scaling and independent lifecycle management.

- **Web UI**
  - Provide a minimal web UI that:
    - Uses SSE or WebSockets for true token-level streaming.
    - Visualizes retrieved documents and distances alongside the answer.