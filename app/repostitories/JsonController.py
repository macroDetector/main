import app.core.globals as g_vars
from multiprocessing import Queue
import os
import json

def read(user: bool, log_queue: Queue):
    json_dir = g_vars.JsonPath
    
    # user 값에 따라 읽어올 파일명을 명확히 지정합니다.
    if user:
        subfolder = "user"
        filename = "user_move.json"
    else:
        subfolder = "move_data"
        filename = "move_data.json"
        
    file_path = os.path.join(json_dir, subfolder, filename)

    try:
        # 파일이 실제로 존재하는지 확인
        if not os.path.exists(file_path):
            if log_queue:
                log_queue.put(f"[경고]: 파일을 찾을 수 없습니다. ({file_path})")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except Exception as e:
        if log_queue:
            log_queue.put(f"[JSON 읽기 오류]: {e}")
        return []