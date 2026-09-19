"""Configuration module for LearnForge Support Assistant.

Loads, validates, and exposes typed application settings from environment
variables and optional .env files. Contains no business logic.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and runtime hyper-parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # LLM Settings (Groq default provider)
    groq_api_key: str = Field(
        default="",
        description="Groq API key for LLM structured output generation.",
    )
    llm_model: str = Field(
        default="openai/gpt-oss-20b",
        description="Default model identifier to use on the Groq platform.",
    )

    # Storage & Indexing Paths
    qdrant_path: str = Field(
        default=".storage/qdrant",
        description="Local filesystem directory for persistent Qdrant index storage.",
    )
    qdrant_collection: str = Field(
        default="learnforge_support",
        description="Qdrant collection name for knowledge records.",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory containing the raw Markdown knowledge base files.",
    )

    # Embedding & Reranking Models
    dense_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed dense text embedding model identifier.",
    )
    sparse_model: str = Field(
        default="Qdrant/bm25",
        description="FastEmbed sparse BM25 text embedding model identifier.",
    )
    rerank_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        description="FastEmbed cross-encoder reranker model identifier.",
    )

    # Retrieval and Fusion Parameters
    dense_top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of candidates retrieved via dense vector search.",
    )
    sparse_top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of candidates retrieved via sparse BM25 search.",
    )
    fused_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of candidates retained after Reciprocal Rank Fusion (RRF).",
    )
    final_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top evidence items retained after cross-encoder reranking.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Smoothing constant for Reciprocal Rank Fusion (standard RRF_K=60).",
    )

    # Conversation & Operational Settings
    max_conversation_turns: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum recent turns retained in bounded conversation history.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for standard output logging.",
    )

    # Observability Settings (Langfuse optional tracing)
    langfuse_enabled: bool = Field(
        default=False,
        description="Whether Langfuse observability tracing is enabled.",
    )
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse project public key.",
    )
    langfuse_secret_key: str = Field(
        default="",
        description="Langfuse project secret key.",
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse API host/base URL.",
    )
    langfuse_tracing_environment: str = Field(
        default="demo",
        description="Langfuse tracing environment identifier.",
    )


@lru_cache
def get_settings() -> Settings:
    """Returns a cached instance of validated application settings."""
    return Settings()
