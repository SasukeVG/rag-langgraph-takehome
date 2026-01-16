from typing import Any, Callable, Iterator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from loguru import logger

from config import settings


def create_llm(
    model: Optional[str] = None,
    temperature: float = 0.4,
    streaming: bool = True,
    timeout: int = 30,
    max_retries: int = 2,
) -> ChatOpenAI:
    api_key = settings.openrouter.api_key
    model_name = model or settings.openrouter.model

    logger.info(f"Initializing OpenRouter LLM with model: {model_name}")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=settings.openrouter.base_url,
        temperature=temperature,
        streaming=streaming,
        timeout=timeout,
        max_retries=max_retries,
    )

    return llm


def stream_response(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
) -> Iterator[str]:
    for chunk in llm.stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            token = chunk.content
            print(token, end="", flush=True)
            yield token
