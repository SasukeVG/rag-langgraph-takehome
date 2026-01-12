from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import settings
from app.graph import RAGGraph
from app.retrieval import DocumentRetriever
from api.schemas import (
    AskRequest,
    AskResponse,
    StatsResponse,
    StepStatus,
    UsedDocument,
)


retriever: Optional[DocumentRetriever] = None
rag_graph: Optional[RAGGraph] = None


def get_retriever() -> DocumentRetriever:
    global retriever
    if retriever is None:
        data_dir = settings.retrieval.data_dir
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        retriever = DocumentRetriever(data_dir=data_dir)
    return retriever


def get_rag_graph() -> RAGGraph:
    global rag_graph
    if rag_graph is None:
        retriever_instance = get_retriever()
        rag_graph = RAGGraph(retriever=retriever_instance)
    return rag_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up API server...")
    try:
        logger.info("Initializing document retriever (loading embeddings model and FAISS)...")
        retriever_instance = get_retriever()
        stats = retriever_instance.get_stats()
        logger.info(
            f"Retriever initialized successfully. Total chunks: {stats.get('total_chunks', 0)}"
        )
    except Exception as e:
        logger.exception(f"Failed to initialize retriever: {e}")
        raise

    logger.info("Initializing RAG graph...")
    try:
        _ = get_rag_graph()
        logger.info("RAG graph initialized successfully")
    except Exception as e:
        logger.exception(f"Failed to initialize RAG graph: {e}")
        raise

    logger.info("API server startup completed")

    yield

    logger.info("Shutting down API server...")


app = FastAPI(
    title="RAG Document API",
    description="API for uploading markdown files to FAISS and retrieving relevant documents",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-document-api"}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    try:
        retriever_instance = get_retriever()
        stats = retriever_instance.get_stats()

        stats["distance_threshold"] = settings.retrieval.distance_threshold

        return StatsResponse(**stats)
    except Exception as e:
        logger.exception(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        graph = get_rag_graph()
        logger.info(f"Processing ask request: {request.query[:50]}...")

        result = graph.invoke(query=request.query, messages=None)

        retrieved_docs = result.get("retrieved_docs", [])
        distances = result.get("distances", [])
        needs_clarification = result.get("needs_clarification", False)
        answer = result.get("answer", "")

        documents_with_scores = []
        if retrieved_docs:
            for doc, score in zip(retrieved_docs, distances):
                source = doc.metadata.get("source", "unknown")
                if source and source != "unknown":
                    filename = Path(source).name if "/" in source or "\\" in source else source
                    documents_with_scores.append(
                        UsedDocument(source=filename, score=round(float(score), 4))
                    )

            documents_with_scores.sort(key=lambda x: x.score)

        steps = []

        steps.append(
            StepStatus(
                step="retrieve",
                status="done",
                docs_found=len(retrieved_docs),
            )
        )
        steps.append(
            StepStatus(
                step="decision",
                status="done",
                clarify=needs_clarification,
            )
        )
        if needs_clarification:
            steps.append(
                StepStatus(
                    step="clarify",
                    status="done",
                )
            )
        else:
            steps.append(
                StepStatus(
                    step="answer",
                    status="done",
                )
            )

        logger.info(
            f"Ask request completed: {len(retrieved_docs)} docs, "
            f"clarify={needs_clarification}, answer_length={len(answer)}, "
            f"documents={len(documents_with_scores)}"
        )

        return AskResponse(
            status="completed",
            steps=steps,
            answer=answer,
            documents=documents_with_scores,
        )

    except Exception as e:
        logger.exception(f"Failed to process ask request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process ask request: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload_server,
    )
