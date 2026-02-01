import queue
from app.core.settings import settings

MOUSE_QUEUE = queue.Queue()


IS_PRESSED = 0
MAX_QUEUE_SIZE = 5000

SEQ_LEN = settings.SEQ_LEN
STRIDE = settings.STRIDE

FEATURES = [
    "speed", "acc", "jerk", "turn", "turn_acc"
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

LOG_QUEUE = None
CHART_DATA = None
TRAIN_DATA = None

save_path = "app/models/weights/mouse_macro_lstm_best.pt"
scaler_path = "app/models/weights/scaler.pkl"

def init_manager():
    global LOG_QUEUE
    global CHART_DATA
    global TRAIN_DATA
    
    from multiprocessing import Manager
    manager = Manager()
    LOG_QUEUE = manager.Queue()
    CHART_DATA = manager.Queue()
    TRAIN_DATA = manager.Queue()