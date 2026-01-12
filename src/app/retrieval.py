from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sentence_transformers import SentenceTransformer

from utils import handle_retrieval_error


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        super().__init__()
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embedding.tolist()


class DocumentRetriever:
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        embedding_model: str = "intfloat/e5-base-v2",
        similarity_top_k: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.embedding_model = embedding_model

        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        self.vector_store: FAISS | None = None
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        try:
            if not self.data_dir.exists():
                logger.warning(
                    f"Data directory {self.data_dir} does not exist. Creating empty vector store."
                )
                self.vector_store = FAISS.from_texts([""], self.embeddings)
                return

            logger.info(f"Loading documents from {self.data_dir}")
            loader = DirectoryLoader(
                str(self.data_dir),
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=False,
            )
            documents = loader.load()

            if not documents:
                logger.warning(f"No documents found in {self.data_dir}")
                self.vector_store = FAISS.from_texts([""], self.embeddings)
                return

            logger.info(f"Loaded {len(documents)} documents")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            splits = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(splits)} chunks")

            logger.info("Creating FAISS vector store...")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            logger.info("Vector store initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize vector store: {e}")
            self.vector_store = FAISS.from_texts([""], self.embeddings)

    @handle_retrieval_error
    def get_relevant_documents(
        self, query: str, k: int | None = None
    ) -> Tuple[List[Document], List[float]]:
        if k is None:
            k = self.similarity_top_k

        if self.vector_store is None:
            logger.warning("Vector store not initialized. Returning empty results.")
            return [], []

        results = self.vector_store.similarity_search_with_score(query, k=k)

        if not results:
            logger.warning(f"No results found for query: {query[:50]}...")
            return [], []

        documents, distances = zip(*results)

        logger.info(
            f"Retrieved {len(documents)} documents for query: {query[:50]}... "
            f"(min distance: {min(distances):.3f})"
        )

        return list(documents), list(distances)

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("No documents provided to add")
            return

        if self.vector_store is None:
            logger.info("Vector store not initialized. Creating new one...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created vector store with {len(documents)} documents")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")

        try:
            self.vector_store.add_documents(splits)
            logger.info(f"Successfully added {len(splits)} chunks to vector store")
        except Exception as e:
            logger.exception(f"Failed to add documents: {e}")
            raise

    def get_stats(self) -> dict:
        if self.vector_store is None:
            return {
                "status": "not_initialized",
                "documents": 0,
                "total_chunks": 0,
                "embedding_model": self.embedding_model,
                "context_top_k": self.similarity_top_k,
            }

        try:
            total_chunks = 0
            documents = 0

            if hasattr(self.vector_store, "index") and self.vector_store.index.ntotal > 0:
                total_chunks = self.vector_store.index.ntotal

                if hasattr(self.vector_store, "docstore") and self.vector_store.docstore:
                    sources = set()
                    for doc_id in self.vector_store.docstore._dict.keys():
                        doc = self.vector_store.docstore.search(doc_id)
                        if doc and hasattr(doc, "metadata") and "source" in doc.metadata:
                            source = doc.metadata["source"]
                            if source and source != "unknown":
                                sources.add(source)
                    documents = len(sources)

                if documents == 0 and self.data_dir.exists():
                    md_files = list(self.data_dir.glob("**/*.md"))
                    documents = len(md_files)

                return {
                    "status": "ready",
                    "documents": documents,
                    "total_chunks": total_chunks,
                    "embedding_model": self.embedding_model,
                    "context_top_k": self.similarity_top_k,
                }

            return {
                "status": "empty",
                "documents": 0,
                "total_chunks": 0,
                "embedding_model": self.embedding_model,
                "context_top_k": self.similarity_top_k,
            }
        except Exception as e:
            logger.exception(f"Failed to get stats: {e}")
            return {
                "status": "error",
                "documents": 0,
                "total_chunks": 0,
                "embedding_model": self.embedding_model,
                "context_top_k": self.similarity_top_k,
                "error": str(e),
            }
