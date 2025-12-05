from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = Field("gpt-4o-mini", env="MODEL")
    max_retries: int = 2