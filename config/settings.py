from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "autonomous_multiagent_engine"
    environment: str = "dev"

    # Infrastructure Endpoints
    ollama_host: str = "http://host.docker.internal:11434"

    # Model Configurations
    specialist_model_name: str = "specialist-pricer"
    neural_network_path: str = "./data/models/deep_neural_network.pth"
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    cross_encoder_model_name: str = "BAAI/bge-reranker-base"

    # Alerting & External APIs
    ntfy_topic: str = "deal-engine-alerts"
    openai_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()