from pynput import mouse
import time
import app.core.globals as g_vars
from datetime import datetime
from multiprocessing import Queue, Event
from app.services.macro_dectector import MacroDetector

def main(stop_event=None, log_queue:Queue=None, chart_Show=True):
    if stop_event is None:
        stop_event = Event()

    # Detector 초기화
    detector = MacroDetector(
        model_path=g_vars.save_path,
        seq_len=g_vars.SEQ_LEN,
        threshold=g_vars.threshold,
        chart_Show=chart_Show,
        stop_event=stop_event
    )

    detector.start_plot_process()

    if log_queue : log_queue.put(f"weight_threshold : {g_vars.weight_threshold}")
    else:
        print(f"weight_threshold : {g_vars.weight_threshold}")

    # 시작 전 카운트다운
    timeinterval = 7

    while timeinterval > 0:
        msg = f"inference 시작까지 count : {timeinterval}"
        if log_queue: log_queue.put(msg)
        else: print(msg)
        
        time.sleep(1)
        timeinterval -= 1
            
    if log_queue:
        log_queue.put("🟢 Macro Detector Running")
        log_queue.put("🚨 데이터 극 초반은 macro로 작동하며 점차 적으로 하락합니다")
    else:
        print("🟢 Macro Detector Running")
        print("🚨 데이터 극 초반은 macro로 작동하며 점차 적으로 하락합니다")
    
    state = {
        'last_ts': time.perf_counter(),
        "lendata": 0,
    }

    def on_move(x, y):     
        now_ts = time.perf_counter()
        delta = now_ts - state['last_ts']

        if delta >= g_vars.tolerance:
            data = {
                'timestamp': datetime.now().isoformat(),
                'x': int(x),
                'y': int(y),
                'deltatime': delta
            }
            state['last_ts'] = now_ts

            g_vars.MOUSE_QUEUE.put(data)

            if state['lendata'] is not None:
                state['lendata'] += 1

                if state['lendata'] <= g_vars.SEQ_LEN * 2:
                    if log_queue:
                        log_queue.put(f"⏳ Data 수집 중... {state['lendata']} / {g_vars.SEQ_LEN * 2}")
                    else:
                        print(f"⏳ Data 수집 중... {state['lendata']} / {g_vars.SEQ_LEN * 2}")
                elif state['lendata'] == g_vars.SEQ_LEN:
                    if log_queue:
                        log_queue.put("✅ Data 수집 완료")
                    else:
                        print("✅ Data 수집 완료")
                    state['lendata'] = None

    listener = mouse.Listener(on_move=on_move)
    listener.start()

    try:
        while not stop_event.is_set():
            if not g_vars.MOUSE_QUEUE.empty():
                step_data = g_vars.MOUSE_QUEUE.get()
                
                # 실제 무거운 추론 작업 수행
                result = detector.push(step_data)

                if result:
                    m_str = result.get('macro_probability', "0%")
                    raw_e = result.get('raw_error', 0.0)

                    if result.get("is_human", True):
                        log_msg = f"{m_str} (err: {raw_e:.4f})"
                    else:
                        log_msg = f"{m_str} (err: {raw_e:.4f}) 🚨"

                    if log_queue:
                        log_queue.put(log_msg)
                    else:
                        print(log_msg)
            else:
                time.sleep(0.001)

    except Exception as e:
        error_msg = f"에러 발생: {e}"
        if log_queue: log_queue.put(error_msg)
        else: print(error_msg)
    finally:
        listener.stop()  # 리스너 안전 종료
        if log_queue:
            log_queue.put("🛑 Macro Detector Stopped")
        else:
            print("🛑 Macro Detector Stopped")
        stop_event.set()