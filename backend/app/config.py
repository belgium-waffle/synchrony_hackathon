"""Runtime configuration. Every secret comes from the environment or the AWS
credential chain — nothing sensitive is ever committed to this repository."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- service -----------------------------------------------------------
    APP_NAME: str = "Synchrony Next-Gen Credit Intelligence"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: Literal["dev", "staging", "prod"] = "dev"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # --- PostgreSQL + pgvector ---------------------------------------------
    # Never inline a password here. Export DATABASE_URL, or let the service fall
    # back to the in-memory policy store for a laptop demo.
    DATABASE_URL: str | None = None
    DB_POOL_SIZE: int = 5
    DB_CONNECT_TIMEOUT_S: int = 5

    # --- embeddings ---------------------------------------------------------
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384          # must match the vector(384) column in sql/001_init.sql
    RETRIEVAL_TOP_K: int = 2

    # --- AWS Bedrock --------------------------------------------------------
    # Credentials resolve through the standard boto3 chain: IAM role, then
    # AWS_PROFILE, then env vars. The service never reads an access key itself.
    AWS_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    BEDROCK_MAX_TOKENS: int = 1500
    BEDROCK_TEMPERATURE: float = 0.0
    BEDROCK_TIMEOUT_S: int = 30
    USE_MOCK_LLM: bool = True         # flip to false once Bedrock access is provisioned

    # --- decisioning policy -------------------------------------------------
    APPROVE_THRESHOLD: int = 720
    REVIEW_THRESHOLD: int = 640
    MODEL_ARTIFACT_PATH: str = "artifacts/stream_a.joblib"
    TRAINING_ROWS: int = Field(default=2000, ge=500)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
