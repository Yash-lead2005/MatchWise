from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = "MatchWise"
    VERSION: str = "1.0.0"
    DATABASE_URL: str = "sqlite:///./matchwise.db"


settings = Settings()