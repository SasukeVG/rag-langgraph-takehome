import sys
from pathlib import Path

import uvicorn

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config import settings
from logging_config import setup_logging

if __name__ == "__main__":
    setup_logging(level=settings.log_level)
    uvicorn.run(
        "api.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload_server,
    )
