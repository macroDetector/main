import os
import json
import sys

class Settings:
    def __init__(self):
        # 기본값 설정 (모델 관련 파라미터)
        self.SEQ_LEN: int = 300
        self.STRIDE: int = 50
        self.tolerance: float = 0.05
        self.JsonPath: str = "./"
        self.threshold: float = 0.8
        self.d_model: int = 128
        self.num_layers: int = 3
        self.dropout: float = 0.3
        self.batch_size: int = 64
        self.lr: float = 0.0005
        self.CLIP_BOUNDS: dict = {}
        self.n_head: int = 4
        self.weight_threshold: float = 1.0
        self.epoch: int = 100
        self.patience: int = 10
        self.weight_decay: float = 0.5
        self.dim_feedforward: int = 128
        self.improvement_val_loss_cut: float = 0.9
        self.chunk_size: int = 50

    @classmethod
    def load_settings(cls):
        inst = cls()
        # 실행 파일(.exe) 또는 스크립트 위치 기준 경로
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        config_dir = os.path.join(base_path, "config")
        config_path = os.path.join(config_dir, "config.json")

        # 1. config 폴더가 없으면 생성
        os.makedirs(config_dir, exist_ok=True)

        # 2. config.json이 없으면 현재 객체의 기본값으로 파일 생성
        if not os.path.exists(config_path):
            # 인스턴스의 모든 속성을 dict로 변환 (기본값 저장)
            default_config = {k: v for k, v in inst.__dict__.items()}
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            print(f"📝 기본 설정 파일 생성 완료: {config_path}")
        
        # 3. 파일이 있으면 읽어서 객체 속성에 덮어쓰기
        else:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                
                for key, value in config_data.items():
                    if hasattr(inst, key):
                        setattr(inst, key, value)
                print(f"✅ 설정 로드 완료: {config_path}")
            except Exception as e:
                print(f"⚠️ 설정 로드 중 오류 발생 (기본값 사용): {e}")

        return inst

    def save(self):
        """실행 중 변경된 설정을 다시 파일로 저장할 때 사용"""
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        config_path = os.path.join(base_path, "config", "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, indent=4, ensure_ascii=False)
            return True
        except Exception:
            return False

settings = Settings.load_settings()