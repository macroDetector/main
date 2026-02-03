import os
import sys
import queue
from app.core.settings import settings

MOUSE_QUEUE = queue.Queue()


IS_PRESSED = 0
MAX_QUEUE_SIZE = 5000

SEQ_LEN = settings.SEQ_LEN
STRIDE = settings.STRIDE

FEATURES = [
    "speed",         # 이동 속도
    "acc",           # 가속도
    "jerk",          # 가속도의 변화 (보간법 판별 핵심 1)
    "turn",          # 회전 각도 (방향 전환)
    "jerk_std",      # jerk의 일정함 (보간법의 수학적 완벽함 포착)
    "straightness"   # 직선도 (함수 기반 곡선 vs 유저의 미세 수정 비교)
]

LAST_EVENT_TS:float = 0.0

MACRO_DETECTOR  = [] 

Recorder = settings.Recorder
JsonPath = settings.JsonPath

threshold = settings.threshold

# model
d_model=settings.d_model
num_layers=settings.num_layers
dropout=settings.dropout
batch_size=settings.batch_size
lr=settings.lr
tolerance=settings.tolerance
CLIP_BOUNDS = settings.CLIP_BOUNDS

LOG_QUEUE = None
CHART_DATA = None
TRAIN_DATA = None

def get_resource_path(relative_path):
    """빌드 환경과 개발 환경의 경로 차이를 해결합니다."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# --- 이제 모든 경로를 이 함수로 감싸세요 ---
save_path = get_resource_path("app/models/weights/mouse_macro_lstm_best.pt")
scaler_path = get_resource_path("app/models/weights/scaler.pkl")

def init_manager():
    global LOG_QUEUE
    global CHART_DATA
    global TRAIN_DATA
    
    from multiprocessing import Manager
    manager = Manager()
    LOG_QUEUE = manager.Queue()
    CHART_DATA = manager.Queue()
    TRAIN_DATA = manager.Queue()