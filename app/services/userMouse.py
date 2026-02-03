from pynput.mouse import Controller

from multiprocessing import Event

import time
from datetime import datetime

import app.core.globals as g_vars
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

    # 간격
    tolerance = g_vars.tolerance

    # 초기값 설정
    start_time = time.perf_counter()
    end_time = time.perf_counter()
    
    while not stop_event.is_set():
        try:
            if end_time - start_time < tolerance:
                
                end_time = time.perf_counter()
                continue

            x, y = mouse_controller.position

            if pre_x is None or pre_y is None:
                pre_x, pre_y = x, y
                start_time = end_time = time.perf_counter()
                continue

            if x == pre_x and y == pre_y:
                start_time = end_time = time.perf_counter()
                continue
            
            # 중요!
            delta = max(0, end_time - start_time - tolerance)

            print(f"x : {x} || y : {y} || delta : {delta}")

            data = {
                'timestamp': datetime.now().isoformat(),
                'x': int(x),
                'y': int(y),
                'deltatime': delta
            }

            pre_x, pre_y = x, y

    
            if record:
                g_vars.MOUSE_QUEUE.put(data)

            if g_vars.MOUSE_QUEUE.qsize() >= g_vars.MAX_QUEUE_SIZE:
                log_queue.put(f"Data 5000개 초과.. 누적 {5000 * i}")
                i += 1
                cunsume_q(record=record, isUser=isUser, log_queue=log_queue)
                log_queue.put("저장 완료 다음 시퀀스 준비")

            end_time = time.perf_counter()
            start_time = time.perf_counter()
        except Exception as e:
            print(e)
            print("🟢 보호 모드 작동")
            time.sleep(1)
            start_time = time.perf_counter()
            end_time = time.perf_counter()            
            pass

    log_queue.put("🛑 Record 종료 신호 발생 남은 데이터 기록 중")

    cunsume_q(record=record, isUser=isUser, log_queue=log_queue)

    log_queue.put("🛑 Record 종료")
    stop_event.set()