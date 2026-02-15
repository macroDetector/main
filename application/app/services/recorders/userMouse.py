from pynput import mouse
import time
from datetime import datetime
from multiprocessing import Queue, Event
import app.core.globals as g_vars
from app.utilites.cunsume_q import cunsume_q

def record_mouse_path(isUser, stop_event=None, record=True, log_queue: Queue = None):
    if stop_event is None:
        stop_event = Event()

    log_queue.put("[Process] 마우스 리스너 기반 경로 생성 시작")
    
    # 상태 유지를 위한 변수들 (클로저 사용을 위해 리스트나 딕셔너리 활용)
    state = {
        'last_ts': time.perf_counter(),
        'i': 1
    }

    def on_move(x, y):
        now_ts = time.perf_counter()
        delta = now_ts - state['last_ts']

        # 설정한 tolerance(예: 0.02s)보다 시간이 더 흘렀을 때만 기록
        # 마우스가 물리적으로 이동한 순간에 이 조건이 체크됨
        if delta >= g_vars.tolerance:
            data = {
                'timestamp': datetime.now().isoformat(),
                'x': int(x),
                'y': int(y),
                'deltatime': delta  # 0.021, 0.033 등 실제 물리적 시간이 찍힘
            }

            state['last_ts'] = now_ts  # 마지막 기록 시점 업데이트

            if record:
                g_vars.MOUSE_QUEUE.put(data)

            # 큐 관리 로직
            if g_vars.MOUSE_QUEUE.qsize() >= g_vars.MAX_QUEUE_SIZE:
                log_queue.put(f"Data {g_vars.MAX_QUEUE_SIZE}개 초과.. 누적 {g_vars.MAX_QUEUE_SIZE * state['i']}")
                state['i'] += 1
                cunsume_q(record=record, isUser=isUser, log_queue=log_queue)
                log_queue.put("저장 완료 다음 시퀀스 준비")

    # 리스너 정의
    listener = mouse.Listener(on_move=on_move)
    listener.start()

    try:
        # stop_event가 발생할 때까지 메인 프로세스는 대기
        while not stop_event.is_set():
            time.sleep(0.1)
    except Exception as e:
        log_queue.put(f"에러 발생: {e}")
    finally:
        listener.stop()  # 리스너 종료
        log_queue.put("🛑 Record 종료 신호 발생 남은 데이터 기록 중")
        cunsume_q(record=record, isUser=isUser, log_queue=log_queue)
        log_queue.put("🛑 Record 종료")
        stop_event.set()