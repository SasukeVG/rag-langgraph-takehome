# Retrieval-Augmented Generation (RAG)

RAG is a technique that enhances LLM responses by retrieving relevant context from external knowledge sources.

## RAG Pipeline

1. **Document Loading**: Load documents from various sources (files, databases, APIs)
2. **Splitting**: Break documents into smaller chunks for efficient retrieval
3. **Embedding**: Convert text chunks into vector representations
4. **Storage**: Store embeddings in a vector database (FAISS, Chroma, Pinecone, etc.)
5. **Retrieval**: Given a query, find the most relevant chunks using similarity search
6. **Generation**: Use retrieved chunks as context when generating the final answer

## Best Practices

- **Chunk size**: Typically 500-1000 characters with 100-200 character overlap
- **Embedding models**: Use semantic embeddings (sentence-transformers, OpenAI embeddings)
- **Retrieval count**: Usually 3-5 most relevant chunks
- **Context relevance**: Filter or rerank retrieved chunks by similarity threshold

## Benefits

- Reduces hallucinations by grounding responses in retrieved context
- Keeps knowledge base up-to-date without retraining models
- Allows using domain-specific documents
- Provides source attribution for answers


