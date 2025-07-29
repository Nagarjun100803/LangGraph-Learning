from pydantic_settings import BaseSettings

class Settings(BaseSettings): 

    tavily_api_key: str
    google_api_key: str
    groq_api_key: str

    langsmith_api_key: str
    langsmith_endpoint: str
    langsmith_tracing: bool
    langsmith_project: str

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
print("API Keys loaded succesfully.")