import os
import sys
import queue
from app.core.settings import settings
from multiprocessing import Lock

lock = Lock()

MOUSE_QUEUE = queue.Queue()


IS_PRESSED = 0
MAX_QUEUE_SIZE = 5000

SEQ_LEN = settings.SEQ_LEN
STRIDE = settings.STRIDE

FEATURES = [
    # 1. 시간적 특성 (Time Dynamics)
    "deltatime",          # 입력 간격
    "dt_cv",              # 시간 간격의 변동성 (기계는 일정함)

    # 2. 이동 물리량 (Magnitude)
    "dist",               # 이동 거리
    "speed",              # 속도
    "acc",                # 가속도
    "jerk",               # 가속도의 변화 (기계와 사람의 가장 큰 차이)

    # 3. 인간 특유의 미세 노이즈 (Jitter)
    "micro_shaking",      # 미세 떨림 (사람은 0이 될 수 없음)
    "jerk_flip_rate",     # 가속도 방향 전환 빈도 (부르르 떨리는 정도)

    # 4. 방향 및 회전 (Directional)
    "turn",               # 회전 각도
    "ang_vel",            # 각속도
    "ang_acc",            # 각가속도 (곡선을 그릴 때의 매끄러움)

    # 5. 경로의 효율성 및 통계 (Statistical)
    "straightness",       # 직선성 (매크로는 완벽한 1.0에 수렴)
    "efficiency_var",     # 경로 효율의 변화량 (인간의 보정 동작 탐지)
    "speed_var",          # 속도 분산
    "acc_smoothness"      # 가속도의 일정함 (매크로 탐지 핵심)
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
n_head=settings.n_head

epoch = settings.epoch
patience = settings.patience
weight_decay = settings.weight_decay
dim_feedforward = settings.dim_feedforward

CLIP_BOUNDS = settings.CLIP_BOUNDS

GLOBAL_CHANGE = False

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