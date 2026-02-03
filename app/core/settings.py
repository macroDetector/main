import os
import json
import sys
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # --- [Section 1] DB 설정 (.env 전용) ---
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "your_db_name"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "your_password"
    
    # --- [Section 2] 앱 및 AI 모델 설정 (config.json 전용) ---
    SEQ_LEN: int = 300
    STRIDE: int = 50
    tolerance: float = 0.05
    JsonPath: str = "./"
    Recorder: str = "json"  # 기본값은 json
    threshold: float = 0.8
    d_model: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    batch_size: int = 64
    lr: float = 0.0005
    CLIP_BOUNDS: dict = {}
    n_head: int = 4

    epoch:int = 100
    patience:int = 10
    weight_decay:float = 0.5
    dim_feedforward:int = 128

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://"
            f"{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True

    @classmethod
    def load_settings(cls):
        # 실행 경로 설정
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        env_path = os.path.join(base_path, ".env")
        config_dir = os.path.join(base_path, "config")
        config_path = os.path.join(config_dir, "config.json")

        # 1. [.env] 파일 자동 생성 (없을 경우)
        if not os.path.exists(env_path):
            env_template = (
                "DB_HOST=localhost\n"
                "DB_PORT=5432\n"
                "DB_NAME=your_db_name\n"
                "DB_USER=postgres\n"
                "DB_PASSWORD=your_password\n"
            )
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_template)
            print("📝 [.env] file created with default templates.")

        # 2. [config/] 폴더 및 [config.json] 기본값 생성
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # 3. JSON 파일 먼저 읽어서 Recorder 확인
        config_data = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"⚠️ config.json 로드 실패: {e}")

        recorder_type = config_data.get("Recorder", "json")

        # 4. Recorder 모드에 따른 로드 처리
        if recorder_type == "postgres":
            inst = cls(_env_file=env_path)
        else:
            inst = cls(_env_file=None) # DB 정보 무시 (기본값 사용)

        # 5. JSON 데이터 병합 (UI 설정값 덮어쓰기)
        for key, value in config_data.items():
            if hasattr(inst, key):
                setattr(inst, key, value)
        
        return inst

# 싱글톤 객체 생성
settings = Settings.load_settings()