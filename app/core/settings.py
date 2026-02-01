from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # DB
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str = "postgres"
    DB_USER: str
    DB_PASSWORD: str
    
    SEQ_LEN : int = 300
    STRIDE : int = 50
    
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://"
            f"{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    JsonPath: str = "./"
    Recorder: str = "postgres"
    threshold: float = 0.8
    d_model:int=128
    num_layers:int=3
    dropout:float=0.3
    batch_size:int=64
    lr:float=0.0005

    class Config:
        env_file = ".env"
        case_sensitive = True
    
settings = Settings()
