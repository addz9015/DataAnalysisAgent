"""API configuration"""
import os

class Settings:
    API_KEY = os.getenv("API_KEY", "dev-key")
    RATE_LIMIT = 60
    LLM_THRESHOLD = 0.6

settings = Settings()