from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Tüm pipeline için merkezi, parametrik konfigürasyon."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    RAW_DATA_DIR: Path = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    FEATURES_DATA_DIR: Path = Path("data/features")

    SAMPLE_RATE: int = 44100
    MAX_CHUNK_DURATION_SEC: int = 10
    SILENCE_THRESH_DB: int = -40
    MIN_SILENCE_LEN_MS: int = 500

    WANDB_PROJECT: str = "ai-voice-pipeline"
    TRAIN_EPOCHS: int = 100
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 0.0001


settings = PipelineSettings()
