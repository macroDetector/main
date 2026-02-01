from pynput.mouse import Controller

from multiprocessing import Event

import time
from datetime import datetime

import app.core.globals as globals
from multiprocessing import Queue

from app.services.cunsume_q import cunsume_q

def record_mouse_path(isUser, stop_event=None, record=True, log_queue:Queue=None):
    if stop_event is None:
        stop_event = Event()

    mouse_controller = Controller()

    log_queue.put("[Process] 마우스 경로 생성 시작")
    i = 1

    pre_x = None
    pre_y = None
    last_ts = time.perf_counter()

    while not stop_event.is_set():
        x, y = mouse_controller.position
        now_ts = time.perf_counter()

        if pre_x is None or pre_y is None:
            pre_x, pre_y = x, y
            last_ts = now_ts
            continue

        if x == pre_x and y == pre_y:
            last_ts = now_ts
            continue

        delta = now_ts - last_ts
        
        last_ts = now_ts

        data = {
            'timestamp': datetime.now().isoformat(),
            'x': int(x),
            'y': int(y),
            'deltatime': delta
        }

        pre_x, pre_y = x, y

   
        if record:
            globals.MOUSE_QUEUE.put(data)

        if globals.MOUSE_QUEUE.qsize() >= globals.MAX_QUEUE_SIZE:
            log_queue.put(f"Data 5000개 초과.. 누적 {5000 * i}")
            i += 1
            cunsume_q(record=record, isUser=isUser, log_queue=log_queue)
            log_queue.put("저장 완료 다음 시퀀스 준비")

    log_queue.put("🛑 Record 종료 신호 발생 남은 데이터 기록 중")

    cunsume_q(record=record, isUser=isUser, log_queue=log_queue)

    log_queue.put("🛑 Record 종료")
    stop_event.set()