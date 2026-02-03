import os
import json

from multiprocessing import Queue
import app.core.globals as g_vars

from app.db.session import SessionLocal
from app.models.MousePoint import MousePoint

def cunsume_q(record:bool, isUser:bool, log_queue:Queue = None):
    all_data = []

    while not g_vars.MOUSE_QUEUE.empty():
        all_data.append(g_vars.MOUSE_QUEUE.get())        
    
    all_data.sort(key=lambda x: x['timestamp'])

    if record and g_vars.Recorder == "postgres":
        db = SessionLocal()
        try:
            for item in all_data:
                mp = MousePoint(
                    timestamp=item['timestamp'], 
                    x=item['x'], 
                    y=item['y'],
                    deltatime=item['deltatime']
                )
                db.add(mp)
            db.commit()
            if log_queue.put:
                log_queue.put(f"[Process] 총 {len(all_data)}개 포인트 DB 저장 완료")
        except Exception as e:
            db.rollback()
            if log_queue.put:
                log_queue.put(f"[Process] DB 저장 오류: {e}")
        finally:
            db.close()
    elif record and g_vars.Recorder == "json":
        if isUser:
            save_dir = os.path.join(g_vars.JsonPath, "user")
            os.makedirs(save_dir, exist_ok=True)            
            file_name = "user_move.json"
        else:
            save_dir = os.path.join(g_vars.JsonPath, "move_data")
            os.makedirs(save_dir, exist_ok=True)              
            file_name = "move_data.json"

        file_path = os.path.join(save_dir, file_name)

        try:
            # 기존 JSON 읽기
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = []
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []

            # 기존 데이터에 새 데이터 추가
            existing_data.extend(all_data)

            # 다시 저장
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4, default=str)

            if log_queue.put:
                log_queue.put(f"[Process] 총 {len(all_data)}개 포인트 JSON 추가 저장 완료: {file_path}")

        except Exception as e:
            if log_queue.put:
                log_queue.put(f"[Process] JSON 저장 오류: {e}")