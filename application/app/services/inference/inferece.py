from pynput import mouse
import time
import app.core.globals as g_vars
from datetime import datetime
from multiprocessing import Queue, Event
from app.services.inference.macro_dectector import MacroDetector
from queue import Empty
from tkinter import filedialog, messagebox
import os
from collections import deque

def main(stop_event=None, log_queue:Queue=None, chart_Show=True):
    use_existing = False
    if g_vars.init_model_path and g_vars.init_scale_path:
        if os.path.exists(g_vars.init_model_path) and os.path.exists(g_vars.init_scale_path):
            model_name = os.path.basename(g_vars.init_model_path)
            msg = f"이전에 사용한 모델을 다시 사용하시겠습니까?\n\n모델: {model_name}"
            use_existing = messagebox.askyesno("경로 재사용", msg)
        else:
            if log_queue: log_queue.put("⚠️ 이전 모델 파일이 경로에 없습니다. 새로 선택합니다.")

    # 2. '아니오'를 눌렀거나 기존 경로가 없는 경우에만 새로 선택
    if not use_existing:
        initial_dir = g_vars.scaler_path
        
        # (1) 모델 선택
        new_model_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="[1/2] 학습된 모델(.pt) 파일을 선택하세요",
            filetypes=(("PyTorch 모델", "*.pt"), ("모든 파일", "*.*"))
        )
        if not new_model_path:
            if log_queue: log_queue.put("❌ 모델 선택이 취소되었습니다.")
            return
        g_vars.init_model_path = new_model_path

        # (2) 스케일러 선택
        new_scale_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="[2/2] 해당 모델의 스케일러(.pkl) 파일을 선택하세요",
            filetypes=(("스케일러 파일", "*.pkl"), ("모든 파일", "*.*"))
        )
        if not new_scale_path:
            if log_queue: log_queue.put("❌ 스케일러 선택이 취소되었습니다.")
            return
        g_vars.init_scale_path = new_scale_path

    # 3. 최종 경로 확정 로그 (이 부분을 g_vars 사용으로 수정!)
    if log_queue:
        # local variable 대신 g_vars 값을 참조하여 에러 방지
        m_name = os.path.basename(g_vars.init_model_path)
        s_name = os.path.basename(g_vars.init_scale_path)
        log_queue.put(f"📂 로드 완료:\n- 모델: {m_name}\n- 스케일러: {s_name}")

    if stop_event is None:
        stop_event = Event()

    # Detector 초기화
    detector = MacroDetector(
        model_path=g_vars.init_model_path,
        seq_len=g_vars.SEQ_LEN,
        threshold=g_vars.threshold,
        chart_Show=chart_Show,
        stop_event=stop_event,
        scale_path=g_vars.init_scale_path
    )

    detector.start_plot_process()

    if log_queue : log_queue.put(f"weight_threshold : {g_vars.weight_threshold}")
    else:
        print(f"weight_threshold : {g_vars.weight_threshold}")

    # 시작 전 카운트다운
    timeinterval = 5

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

    data_queue = Queue()

    def on_move(x, y):
        nonlocal data_queue
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

            data_queue.put(data)

            if state['lendata'] is not None:
                state['lendata'] += 1

                if state['lendata'] <= detector.allowable_add_data:
                    if log_queue:
                        log_queue.put(f"⏳ Data 수집 중... {state['lendata']} / {detector.allowable_add_data}")
                    else:
                        print(f"⏳ Data 수집 중... {state['lendata']} / {detector.allowable_add_data}")
                elif state['lendata'] == detector.allowable_add_data:
                    if log_queue:
                        log_queue.put("✅ Data 수집 완료")
                    else:
                        print("✅ Data 수집 완료")
                    state['lendata'] = None

    listener = mouse.Listener(on_move=on_move)
    listener.start()

    junk_buffer = deque(maxlen=detector.allowable_add_data)
    try:
        while not stop_event.is_set():
            
            try:
                data = data_queue.get(timeout=0.05)
                junk_buffer.append(data)
                
                while not data_queue.empty():
                    junk_buffer.append(data_queue.get_nowait())
            except Empty:
                continue
            
            if len(junk_buffer) >= detector.allowable_add_data:
                for data in junk_buffer:
                    result = detector.push(data)

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
                junk_buffer.clear()

    except Exception as e:
        error_msg = f"에러 발생: {e}"
        if log_queue: log_queue.put(error_msg)
        else: print(error_msg)
    finally:
        detector.buffer.clear()
        listener.stop()  # 리스너 안전 종료
        if log_queue:
            log_queue.put("🛑 Macro Detector Stopped")
        else:
            print("🛑 Macro Detector Stopped")
        try:
            while True:
                g_vars.CHART_DATA.get_nowait()
        except Empty:
            pass            
        stop_event.set()