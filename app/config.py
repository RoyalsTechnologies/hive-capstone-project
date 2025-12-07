"""
Configuration settings for the application
"""

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # CORS Settings
    # SECURITY: In production, replace ["*"] with specific allowed origins
    # Example: ["https://yourdomain.com", "https://www.yourdomain.com"]
    CORS_ORIGINS: List[str] = ["*"]  # WARNING: Allows all origins - restrict in production

    # Server Settings
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
