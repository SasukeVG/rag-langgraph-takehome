from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from loguru import logger

from config import settings
from app.llm import create_llm, stream_response
from app.retrieval import DocumentRetriever
from utils import handle_graph_execution_error, retry_with_backoff


def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    return left + right


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_docs: list[Document]
    distances: list[float]
    needs_clarification: bool
    answer: str


class RAGGraph:
    def __init__(
        self,
        retriever: DocumentRetriever,
        distance_threshold: Optional[float] = None,
    ):
        self.retriever = retriever
        self.distance_threshold = (
            distance_threshold
            if distance_threshold is not None
            else settings.retrieval.distance_threshold
        )
        self.llm = create_llm(streaming=True)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("decision", self.decision_node)
        workflow.add_node("answer", self.answer_node)
        workflow.add_node("clarify", self.clarify_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "decision")
        workflow.add_conditional_edges(
            "decision",
            self.should_clarify,
            {
                "answer": "answer",
                "clarify": "clarify",
            },
        )
        workflow.add_edge("answer", END)
        workflow.add_edge("clarify", END)

        return workflow.compile()

    def retrieve_node(self, state: GraphState) -> GraphState:
        query = state.get("query", "")
        if not query:
            logger.warning("Empty query in retrieve_node")
            return {
                **state,
                "retrieved_docs": [],
                "distances": [],
            }

        logger.info(f"[Retrieve] Processing query: {query[:50]}...")
        documents, distances = self.retriever.get_relevant_documents(query, k=5)

        return {
            **state,
            "retrieved_docs": documents,
            "distances": distances,
        }

    def decision_node(self, state: GraphState) -> GraphState:
        distances = state.get("distances", [])

        if not distances:
            logger.warning("[Decision] No distances available. Asking clarification.")
            return {
                **state,
                "needs_clarification": True,
            }

        min_distance = min(distances)
        needs_clarification = min_distance > self.distance_threshold

        logger.info(
            f"[Decision] min distance: {min_distance:.3f}, "
            f"threshold: {self.distance_threshold:.3f}, "
            f"clarify: {needs_clarification}"
        )

        return {
            **state,
            "needs_clarification": needs_clarification,
        }

    def should_clarify(self, state: GraphState) -> Literal["answer", "clarify"]:
        if state.get("needs_clarification", False):
            return "clarify"
        return "answer"

    @retry_with_backoff(max_retries=2, backoff_factor=1.5, exceptions=(Exception,))
    def answer_node(self, state: GraphState) -> GraphState:
        query = state.get("query", "")
        documents = state.get("retrieved_docs", [])
        distances = state.get("distances", [])
        messages = state.get("messages", [])

        cutoff = self.distance_threshold
        filtered_docs = [doc for doc, dist in zip(documents, distances) if dist <= cutoff]

        logger.info(
            f"[Answer] Using {len(filtered_docs)}/{len(documents)} documents "
            f"(filtered by distance threshold: {cutoff:.3f})"
        )

        context_parts = []
        for i, doc in enumerate(filtered_docs, 1):
            source = doc.metadata.get("source", "unknown")
            filename = source.split("/")[-1] if "/" in source else source
            context_parts.append(f"--- Document {i} ({filename}) ---\n{doc.page_content}\n")

        context = "\n".join(context_parts)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that answers questions strictly based on the provided context documents.
Do not add information that is not explicitly supported by the context.
When answering, cite the source documents when relevant (e.g., "Source: docX.md").
If the context does not contain enough information to fully answer the question,
state this explicitly instead of filling gaps from general knowledge.

Context documents:
{context}

Use the conversation history to understand the context of follow-up questions.""",
                ),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{query}"),
            ]
        )

        history_messages = messages if messages else []

        formatted_messages = prompt_template.format_messages(
            context=context,
            messages=history_messages,
            query=query,
        )

        print("\n[assistant] ", end="", flush=True)
        answer_tokens = []
        for token in stream_response(self.llm, formatted_messages):
            answer_tokens.append(token)
        answer = "".join(answer_tokens).strip()

        logger.info(f"[Answer] Generated response ({len(answer)} characters)")

        return {
            **state,
            "answer": answer,
            "messages": messages
            + [
                HumanMessage(content=query),
                AIMessage(content=answer),
            ],
        }

    @retry_with_backoff(max_retries=2, backoff_factor=1.5, exceptions=(Exception,))
    def clarify_node(self, state: GraphState) -> GraphState:
        query = state.get("query", "")
        messages = state.get("messages", [])
        distances = state.get("distances", [])
        min_distance = min(distances) if distances else 1.0

        logger.info("[Clarify] Generating clarification question...")

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant. The user asked a question, but the available context 
documents are not highly relevant (distance: {distance:.3f}, lower is better). 

Generate a friendly clarification question to help the user refine their query. 
The question should be specific and guide them to provide more context or rephrase their question.

Original question: {query}""",
                ),
            ]
        )

        formatted_messages = prompt_template.format_messages(
            distance=min_distance,
            query=query,
        )

        print("\n[assistant] ", end="", flush=True)
        response = self.llm.invoke(formatted_messages)
        clarification = response.content.strip()
        print(clarification)

        logger.info(f"[Clarify] Generated clarification question")

        return {
            **state,
            "answer": clarification,
            "messages": messages
            + [
                HumanMessage(content=query),
                AIMessage(content=clarification),
            ],
        }

    @handle_graph_execution_error
    def invoke(self, query: str, messages: list[BaseMessage] | None = None) -> GraphState:
        initial_state: GraphState = {
            "messages": messages or [],
            "query": query,
            "retrieved_docs": [],
            "distances": [],
            "needs_clarification": False,
            "answer": "",
        }

        result = self.graph.invoke(initial_state)
        return result
