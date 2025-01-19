from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent


class Config(BaseSettings):
    """Application settings"""

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "de15-jd"

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimension: int = 3072

    # Anthropic
    anthropic_api_key: str
    anthropic_chat_model: str = "claude-3-5-sonnet-latest"

    # Application Configuration
    app_page_title: str = "Jobs AI"
    app_page_icon: str = "🤖"

    class Config:
        env_file = str(ROOT_DIR / ".env")


@lru_cache()
def get_config() -> Config:
    """Get cached settings"""
    return Config()