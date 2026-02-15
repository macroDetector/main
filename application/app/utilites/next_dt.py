import time
import app.core.globals as g_vars

def next_dt():
    now = time.perf_counter()
    if g_vars.LAST_EVENT_TS is None:
        dt = 0.0
    else:
        dt = now - g_vars.LAST_EVENT_TS
    g_vars.LAST_EVENT_TS = now
    return dt