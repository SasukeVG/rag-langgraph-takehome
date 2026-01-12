# Vector Databases for RAG

Vector databases enable efficient similarity search over embeddings.

## FAISS (Facebook AI Similarity Search)

**Characteristics:**
- Open-source library by Meta
- Optimized for similarity search and clustering
- Can run entirely in-memory or persist to disk
- Fast and efficient for medium-sized datasets (< 10M vectors)

**Use cases:**
- Local development and prototyping
- Applications that don't require distributed storage
- When you need fast, CPU-based similarity search

## Chroma

**Characteristics:**
- Embeddings-first database
- Simple Python API
- Can run locally or as a server
- Built-in collection management

**Use cases:**
- Quick prototyping
- When you need persistence out of the box
- Applications requiring collection-based organization

## Choosing Between Them

- **FAISS**: Better for in-memory, high-performance search
- **Chroma**: Better for production persistence and simpler API

For this demo, FAISS provides the fastest local development experience.


