import enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE_PATH, override=False)


class LogLevel(str, enum.Enum):
    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        extra="ignore",
    )

    api_key: str = Field(
        default="paste-your-openrouter-api-key-here",
        description="OpenRouter API key",
    )
    model: str = Field(
        default="mistralai/devstral-2512:free",
        description="OpenRouter model to use",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL",
    )


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        extra="ignore",
    )

    data_dir: str = Field(
        default="data",
        description="Directory for document data",
    )
    distance_threshold: float = Field(
        default=0.9,
        description="Distance threshold for retrieval",
    )

    @property
    def data_dir_path(self) -> Path:
        return Path(self.data_dir)


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_",
        extra="ignore",
    )

    host: str = Field(
        default="0.0.0.0",
        description="Host for FastAPI server",
    )
    port: int = Field(
        default=8008,
        description="Port for FastAPI server",
    )
    reload_server: bool = Field(
        default=True,
        description="Reload server when code changes",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    service_name: str = Field(default="tz_spotware", description="Service name")
    environment: str = Field(default="dev", description="Current environment (dev, prod, etc.)")
    log_level: str = Field(default="DEBUG", description="Logging level")

    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Base directory of the project",
    )

    openrouter: OpenRouterSettings = OpenRouterSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    api: APISettings = APISettings()

    @property
    def openrouter_base_url(self) -> str:
        return self.openrouter.base_url


settings = Settings()  # type: ignore
