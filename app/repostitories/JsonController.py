import app.core.globals as globals
from multiprocessing import Queue

def read(user: bool, log_queue:Queue):
    import os, json

    json_dir = globals.JsonPath
    subfolder = "user" if user else "macro"
    path = os.path.join(json_dir, subfolder)

    try:
        if not os.path.exists(path):
            return []  # 폴더 없으면 빈 리스트

        files = [f for f in os.listdir(path) if f.endswith(".json")]
        if not files:
            return []  # 파일 없으면 빈 리스트

        # 파일 하나만 있다고 가정, 첫 번째 파일 사용
        file_path = os.path.join(path, files[0])

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except Exception as e: 
        if log_queue:
            log_queue.put(f"[JSON 읽기 오류]: {e}")
        return []
