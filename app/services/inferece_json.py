import time
import app.core.globals as g_vars
from datetime import datetime
from multiprocessing import Queue

from app.services.macro_dectector import MacroDetector
from multiprocessing import Event
from app.repostitories.JsonController import read

def main(stop_event=None, log_queue:Queue=None, chart_Show=True):
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

    timeinterval = 7

    while timeinterval != 0:
        timeinterval -= 1
        log_queue.put(f"train 시작까지 count : {timeinterval}")

        time.sleep(1)

    if log_queue:
        log_queue.put("🟢 Macro Detector Running")
    else:
        print("🟢 Macro Detector Running")

    user_data:list[dict] = read(user=False, log_queue=log_queue)

    print(user_data[:5])

    for step in user_data:
        if stop_event.is_set():
            log_queue.put("🛑 Detector 중지")
            break
        data = {
            'timestamp': datetime.fromisoformat(step.get("timestamp")),
            'x': step.get("x"),
            'y': step.get("y"),
            'deltatime': step.get("deltatime")  
        }
        result = detector.push(data)

        if result:
            # 확률 수치(float)를 가져옵니다.
            m_prob = result.get('prob_value', 0.0) 
            m_str = result.get('macro_probability', "0%")
            raw_e = result.get('raw_error', 0.0)

            if result["is_human"]:
                log_msg = f"🙂 HUMAN | {m_str} (err: {raw_e:.4f})"
            else:
                # 매크로 판정 시 사이렌 이모지와 함께 확률 강조
                log_msg = f"🚨 MACRO DETECTED | {m_str} (err: {raw_e:.4f}) 🚨"

            # 출력 대상 선택 (Queue 혹은 Print)
            if log_queue:
                log_queue.put(log_msg)
            else:
                print(log_msg)

    try:
        # stop_event가 발생할 때까지 메인 프로세스는 대기
        while not stop_event.is_set():
            time.sleep(0.1)
    except Exception as e:
        log_queue.put(f"에러 발생: {e}")
    finally:
        log_queue.put("🛑 Detector 종료")
        stop_event.set()

    if log_queue:
        log_queue.put("🛑 Macro Detector Stopped")
    else:
        print("🛑 Macro Detector Stopped")

    stop_event.set()