import os
import json
import datetime
import tkinter as tk
from tkinter import simpledialog

from multiprocessing import Queue
import app.core.globals as g_vars
from app.models.MousePoint import MousePoint

def get_filename_from_user(default_name):
    root = tk.Tk()
    root.withdraw()

    filename = simpledialog.askstring(
        "파일 이름 입력",
        "저장할 파일 이름을 입력하세요:",
        initialvalue=default_name
    )

    root.destroy()
    return filename

def cunsume_q(record: bool, isUser: bool, log_queue: Queue = None):
    all_data = []

    while not g_vars.MOUSE_QUEUE.empty():
        all_data.append(g_vars.MOUSE_QUEUE.get())

    all_data.sort(key=lambda x: x['timestamp'])

    subfolder = "user" if isUser else "move_data"
    base_name = "user_move" if isUser else "move_data"
    save_dir = os.path.join(g_vars.JsonPath, subfolder)

    os.makedirs(save_dir, exist_ok=True)

    # 🔥 사용자 입력 받기
    user_input_name = get_filename_from_user(base_name)

    # 취소 눌렀을 경우
    if not user_input_name:
        user_input_name = base_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path = os.path.join(save_dir, f"{user_input_name}.json")

    # 같은 이름 존재하면 자동 번호 붙이기
    counter = 1
    original_path = file_path
    while os.path.exists(file_path):
        file_path = original_path.replace(".json", f"_{counter}.json")
        counter += 1

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4, default=str)

        if log_queue is not None:
            log_queue.put(f"[Process] 저장 완료 ({len(all_data)}개): {file_path}")

    except Exception as e:
        if log_queue is not None:
            log_queue.put(f"[Process] JSON 저장 오류: {e}")
