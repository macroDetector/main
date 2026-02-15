import app.core.globals as g_vars
from multiprocessing import Queue
import os
import json
from tkinter import filedialog, messagebox

def read(log_queue: Queue = None):
    json_dir = g_vars.JsonPath
    
    # 1. 탐색기 창 열기 (선택된 파일의 전체 경로가 return_path에 담김)
    full_path = filedialog.askopenfilename(
        initialdir=json_dir,
        title="읽어올 JSON 파일을 선택하세요",
        filetypes=(("JSON 파일", "*.json"), ("모든 파일", "*.*"))
    )

    # 2. 선택 취소 처리
    if not full_path:
        if log_queue:
            log_queue.put("[알림]: 파일 선택이 취소되었습니다.")
        return None, [] # (파일명, 데이터) 형태로 리턴하면 관리하기 편함

    # 3. 전체 경로에서 파일 이름만 쏙 뽑아오기
    # 예: "C:/data/user/move_01.json" -> "move_01.json"
    raw_filename = os.path.basename(full_path)
    filename = os.path.splitext(raw_filename)[0]

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if log_queue:
            log_queue.put(f"[성공]: '{filename}' 파일을 불러왔습니다.")
            
        return filename, data

    except Exception as e:
        if log_queue:
            log_queue.put(f"[JSON 읽기 오류]: {e}")
        return None, []