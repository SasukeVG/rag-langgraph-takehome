import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from loguru import logger

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 2,
    backoff_factor: float = 1.5,
    initial_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.exception(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception  # type: ignore

        return cast(Callable[..., T], wrapper)

    return decorator


def handle_retrieval_error(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Retrieval error in {func.__name__}: {e}")
            return ([], [])

    return wrapper


def handle_llm_error(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"LLM error in {func.__name__}: {e}")
            return (
                "I apologize, but I encountered an error while processing your request. "
                "Please try again or rephrase your question."
            )

    return wrapper


def handle_graph_execution_error(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Graph execution error in {func.__name__}: {e}")
            return {
                "messages": kwargs.get("messages", []),
                "query": kwargs.get("query", ""),
                "retrieved_docs": [],
                "distances": [],
                "needs_clarification": False,
                "answer": (
                    "I encountered an error while processing your query. "
                    "Please try again or contact support if the issue persists."
                ),
            }

    return wrapper
