from pynput.mouse import Controller

import time
import app.core.globals as globals
from datetime import datetime
from multiprocessing import Queue

from app.services.macro_dectector import MacroDetector
from multiprocessing import Event

def main(stop_event=None, log_queue:Queue=None, chart_Show=True):
    if stop_event is None:
        stop_event = Event()

    detector = MacroDetector(
        model_path=globals.save_path,
        seq_len=globals.SEQ_LEN,
        threshold=globals.threshold,
        chart_Show=chart_Show,
        stop_event=stop_event
    )

    detector.start_plot_process()
    
    if log_queue:
        log_queue.put("🟢 Macro Detector Running")
    else:
        print("🟢 Macro Detector Running")

    mouse_controller = Controller()

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

        result = detector.push(data)

        if result:
            if result["is_human"]:
                if log_queue:
                    log_queue.put(f"🙂 HUMAN | prob={result['prob']:.3f}")
                else:
                    print(f"🙂 HUMAN | prob={result['prob']:.3f}")
            else:
                if log_queue:
                    log_queue.put(f"🚨 MACRO | prob={result['prob']:.3f}") 
                else:
                    print(f"🚨 MACRO | prob={result['prob']:.3f}") 
    if log_queue:
        log_queue.put("🛑 Macro Detector Stopped")
    else:
        print("🛑 Macro Detector Stopped")

    stop_event.set()    