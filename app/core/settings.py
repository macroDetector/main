import os
import json
import sys
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "your_db_name"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "your_password"

    SEQ_LEN: int = 300
    STRIDE: int = 50
    tolerance: float = 0.05
    JsonPath: str = "./"
    Recorder: str = "json"
    threshold: float = 0.8
    d_model: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    batch_size: int = 64
    lr: float = 0.0005
    CLIP_BOUNDS: dict = {}
    n_head: int = 4

    epoch: int = 100
    patience: int = 10
    weight_decay: float = 0.5
    dim_feedforward: int = 128

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
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))

        env_path = os.path.join(base_path, ".env")
        config_dir = os.path.join(base_path, "config")
        config_path = os.path.join(config_dir, "config.json")


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
            print("📝 [.env] file created (default template).")

        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        if not os.path.exists(config_path):
            default_config = {
                "SEQ_LEN": 100,
                "STRIDE": 50,
                "JsonPath": "./",
                "Recorder": "json",
                "threshold": 0.0,
                "d_model": 256,
                "num_layers": 3,
                "dropout": 0.3,
                "batch_size": 128,
                "lr": 0.0005,
                "tolerance": 0.02,
                "n_head": 8,
                "epoch": 70,
                "patience": 10,
                "weight_decay": 0.2,
                "dim_feedforward": 512,
                "CLIP_BOUNDS": {
                    "deltatime": {"min": 0.0, "max": 0.0},
                    "dt_cv": {"min": 0.0, "max": 0.0},
                    "dist": {"min": 0.0, "max": 0.0},
                    "speed": {"min": 0.0, "max": 0.0},
                    "acc": {"min": 0.0, "max": 0.0},
                    "jerk": {"min": 0.0, "max": 0.0},
                    "micro_shaking": {"min": 0.0, "max": 0.0},
                    "jerk_flip_rate": {"min": 0.3, "max": 0.9},
                    "turn": {"min": 0.0, "max": 0.0},
                    "ang_vel": {"min": 0.0, "max": 0.0},
                    "ang_acc": {"min": 0.0, "max": 0.0},
                    "straightness": {"min": 0.0, "max": 0.0},
                    "efficiency_var": {"min": 0.0, "max": 0.0},
                    "speed_var": {"min": 0.0, "max": 0.0},
                    "acc_smoothness": {"min": 0.0, "max": 0.0}
                },
                "drop_out": 0.3
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)

            print("📝 [config/config.json] default file created.")

        config_data = {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"⚠️ config.json load failed: {e}")

        recorder_type = config_data.get("Recorder", "json")

        if recorder_type == "postgres":
            inst = cls(_env_file=env_path)
        else:
            inst = cls(_env_file=None)

        for key, value in config_data.items():
            if hasattr(inst, key):
                setattr(inst, key, value)

        return inst

settings = Settings.load_settings()
