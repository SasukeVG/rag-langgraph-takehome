import sys
from typing import Optional

from loguru import logger

_logging_configured = False


def setup_logging(level: Optional[str] = None) -> None:
    global _logging_configured

    if _logging_configured:
        return

    if level is None:
        level = "INFO"

    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    logger.disable("httpx")
    logger.disable("httpcore")
    logger.disable("urllib3")
    logger.enable("app")

    _logging_configured = True
    logger.info(f"Logging configured with level: {level}")
