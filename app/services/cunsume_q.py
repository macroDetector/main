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
        import datetime
        
        # 1. 폴더 및 기본 파일명 설정
        subfolder = "user" if isUser else "move_data"
        base_name = "user_move" if isUser else "move_data"
        save_dir = os.path.join(g_vars.JsonPath, subfolder)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 기본 파일 경로
        file_path = os.path.join(save_dir, f"{base_name}.json")

        # 2. 파일이 이미 존재하면 타임스탬프를 붙여서 경로 변경 (새로 만들기)
        if os.path.exists(file_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(save_dir, f"{base_name}_{timestamp}.json")

        try:
            # 3. 합치지 않고 all_data만 바로 저장
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4, default=str)

            if log_queue is not None:
                status_msg = "신규 생성 완료"
                log_queue.put(f"[Process] {status_msg} ({len(all_data)}개): {file_path}")

        except Exception as e:
            if log_queue is not None:
                log_queue.put(f"[Process] JSON 저장 오류: {e}")