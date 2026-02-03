from pynput import mouse
import time
from datetime import datetime
from multiprocessing import Queue, Event
import app.core.globals as g_vars
from app.services.macro_dectector import MacroDetector

def main(stop_event=None, log_queue: Queue = None, chart_Show=True):
    if stop_event is None:
        stop_event = Event()

    detector = MacroDetector(
        model_path=g_vars.save_path,
        seq_len=g_vars.SEQ_LEN,
        threshold=g_vars.threshold,
        chart_Show=chart_Show,
        stop_event=stop_event
    )

    detector.start_plot_process()
    
    msg = "🟢 Macro Detector (Listener Mode) Running"
    if log_queue: log_queue.put(msg)
    else: print(msg)

    # 상태 관리를 위한 딕셔너리
    state = {
        'last_ts': time.perf_counter(),
        'error_start_time': None
    }

    def on_move(x, y):
        try:
            now_ts = time.perf_counter()
            delta = now_ts - state['last_ts']

            # 설정한 tolerance(0.02s)보다 실제 이동 간격이 클 때만 탐지기에 푸시
            if delta >= g_vars.tolerance:
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'x': int(x),
                    'y': int(y),
                    'deltatime': delta # 실제 물리적 시간 (0.0209... 등)
                }

                state['last_ts'] = now_ts
                state['error_start_time'] = None # 정상 작동 시 에러 시간 초기화

                result = detector.push(data)

                if result:
                    m_prob = result.get('prob_value', 0.0) 
                    m_str = result.get('macro_probability', "0%")
                    raw_e = result.get('raw_error', 0.0)

                    if result["is_human"]:
                        log_msg = f"🙂 HUMAN | {m_prob} | {m_str} (err: {raw_e:.4f})"
                    else:
                        log_msg = f"🚨 MACRO DETECTED | {m_str} (err: {raw_e:.4f}) 🚨"

                    if log_queue: log_queue.put(log_msg)
                    else: print(log_msg)

        except Exception as e:
            if state['error_start_time'] is None:
                state['error_start_time'] = time.perf_counter()
                print(f"\n🚨 보호 모드 진입 (원인: {e})")
            
            # 리스너 내부에서는 스레드 안전을 위해 간단한 에러 출력만 권장
            print(f"🟢 감지 중단됨... {e}", end="\r")

    # 리스너 시작
    listener = mouse.Listener(on_move=on_move)
    listener.start()

    try:
        # stop_event가 발생할 때까지 메인 스레드는 대기
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        listener.stop()
        msg = "🛑 Macro Detector Stopped"
        if log_queue: log_queue.put(msg)
        else: print(msg)
        stop_event.set()