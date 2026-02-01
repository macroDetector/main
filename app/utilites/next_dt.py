import time
import app.core.globals as globals

def next_dt():
    now = time.perf_counter()
    if globals.LAST_EVENT_TS is None:
        dt = 0.0
    else:
        dt = now - globals.LAST_EVENT_TS
    globals.LAST_EVENT_TS = now
    return dt