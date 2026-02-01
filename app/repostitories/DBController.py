# sqlalchemy
# 앱 내부
from app.db.session import SessionLocal

from sqlalchemy import text
from app.models.MousePoint import MousePoint
import app.core.globals as globals
from multiprocessing import Queue

def point_clear(log_queue:Queue=None):
    db = SessionLocal()
    try:
        db.query(MousePoint).delete()
        db.commit()

        db.execute(text("TRUNCATE TABLE public.mouse_points RESTART IDENTITY;"))
        db.commit()

        if log_queue:
            log_queue.put("MousePoint 테이블 초기화 완료")
    except Exception as e:
        db.rollback()
        if log_queue:
            log_queue.put(f"테이블 초기화 오류: {e}")
    finally:
        db.close()

def read(user, log_queue:Queue=None):
    db = SessionLocal()
    try:
        if user:
            point = db.query(MousePoint).all()
            
        return point
    except Exception as e:
        if log_queue:
            log_queue.put(f"DB 읽기 오류: {e}")
        return []
    finally:
        db.close()