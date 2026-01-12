from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    k: int = Field(default=3, ge=1, le=20, description="Number of documents to retrieve")


class DocumentResult(BaseModel):
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source file name")
    similarity: float = Field(..., description="L2 distance score")


class QueryResponse(BaseModel):
    query: str
    results: List[DocumentResult]
    total_found: int


class UploadResponse(BaseModel):
    filename: str
    status: str
    chunks_added: int
    message: str


class StatsResponse(BaseModel):
    status: str = Field(
        ..., description="Status of the vector store (ready, empty, not_initialized, error)"
    )
    documents: int = Field(..., description="Number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    embedding_model: str = Field(..., description="Embedding model name")
    distance_threshold: float = Field(..., description="Distance threshold")
    context_top_k: int = Field(..., description="Number of top documents to retrieve")


class AskRequest(BaseModel):
    query: str = Field(..., description="Question to ask the RAG system")

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v


class StepStatus(BaseModel):
    step: str = Field(..., description="Step name")
    status: str = Field(..., description="Step status")
    docs_found: Optional[int] = Field(None, description="Number of documents found")
    clarify: Optional[bool] = Field(None, description="Whether clarification is needed")


class UsedDocument(BaseModel):
    source: str = Field(..., description="Source file name")
    score: float = Field(..., description="Distance score")


class AskResponse(BaseModel):
    status: str = Field(..., description="Overall status")
    steps: List[StepStatus] = Field(..., description="List of pipeline steps")
    answer: str = Field(..., description="Final answer from the RAG system")
    documents: List[UsedDocument] = Field(default_factory=list, description="List of documents")
