import random
import time

from multiprocessing import Event
from multiprocessing import Queue

import pyautogui

from app.repostitories.DBController import read as postgresread
from app.repostitories.JsonController import read as jsonread

import app.core.globals as globals
from app.utilites.make_df_from_points import make_df_from_points

from app.utilites.catmull_rom_spline import linear, ease_in_out_s_curve, ease_in_out_quad_random


pyautogui.FAILSAFE = True
screen_width, screen_height = pyautogui.size()

def copy_move(stop_event=None, log_queue: Queue = None):
    if stop_event is None:
        stop_event = Event()

    # ---------- 데이터 로드 ----------
    if globals.Recorder == "postgres":
        points = postgresread(user=True, log_queue=log_queue)
        is_dict = False
    else:
        points = jsonread(user=True, log_queue=log_queue)
        is_dict = True

    df = make_df_from_points(points, is_dict=is_dict)

    if len(df) < 4:
        if log_queue:
            log_queue.put("❌ 포인트 부족 (최소 4개 필요)")
        return

    if log_queue:
        log_queue.put("🟢 Copy Move 시작")

    print(df)

    first = df.iloc[0]
    pyautogui.moveTo(first['x'], first['y'], duration=0)

    while not stop_event.is_set():       
        for i in range(1, len(df)):
            if stop_event.is_set():
                break

            p0 = (df.iloc[i-1]['x'], df.iloc[i-1]['y'])
            p1 = (df.iloc[i]['x'], df.iloc[i]['y'])
            deltatime = df.iloc[i]['deltatime']

            # 이동 스텝 수: 1px당 1~3 스텝 정도, 또는 최소 5~20 스텝
            steps = max(int(deltatime / 0.00001), 5)  # 예: 델타타임 비례
            pattern = random.choice(['linear', 's_curve', 'ease'])

            for s in range(steps):
                t = s / steps

                # 보간 방식 선택
                if pattern == 'linear':
                    t_mod = linear(t)
                elif pattern == 's_curve':
                    t_mod = ease_in_out_s_curve(t)
                else:
                    t_mod = ease_in_out_quad_random(t)

                # 선형 보간
                x = p0[0] + (p1[0] - p0[0]) * t_mod
                y = p0[1] + (p1[1] - p0[1]) * t_mod

                # 랜덤 미세 노이즈
                if random.random() < 0.05:
                    x += random.randint(-1, 1)
                    y += random.randint(-1, 1)

                x = max(0, min(screen_width - 1, int(x)))
                y = max(0, min(screen_height - 1, int(y)))

                # 실제 이동, duration=0으로 빠르게 이동
                pyautogui.moveTo(x, y, duration=0)
        
        break

    if log_queue:
        log_queue.put("🛑 Copy Move 종료")

    stop_event.set()
