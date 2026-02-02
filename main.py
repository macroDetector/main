import sys
import os

# 1. 가장 먼저 보이는 메시지
print("🚀 프로그램 실행 중... 잠시만 기다려 주세요.")

# 2. 가장 무거운 torch 로딩 시각화
print("📦 라이브러리 로드 중 (PyTorch)...", end="\r")
import torch
print("📦 라이브러리 로드 완료 (PyTorch)   ")

print("⚙️ 시스템 환경 설정 중...", end="\r")
import multiprocessing 
import app.core.globals as globals
import ctypes
from app.core.settings import settings
print("⚙️ 시스템 환경 설정 완료         ")

if __name__ == "__main__":
    multiprocessing.freeze_support() 

    print("Welcome")
    user_input = input("inference Mode? (y/n): ").lower()
    
    inference_Mode = user_input == 'y'

    if inference_Mode:
        import keyboard
        import app.services.inferece as inference
        from multiprocessing import Event

        user_input2 = input("chart Show? (y/n): ").lower()
        chart_Show = user_input2 == 'y'

        ctypes.windll.kernel32.SetConsoleTitleW("Inference Mode (Quit: CTRL+SHIFT+Q)")

        globals.init_manager()

        def trigger_stop_event():
            stop_move_event.set()
            print("🛑 STOP SIGNAL RECEIVED (CTRL+SHIFT+Q)")

        stop_move_event = Event()
        keyboard.add_hotkey('ctrl+shift+q', trigger_stop_event)
        
        inference.main(
            stop_event=stop_move_event,
            chart_Show=chart_Show
        )
    else:
        from app.gui.main_window import VantageUI
        from app.db.session import init_db
        from PyQt6.QtWidgets import QApplication

        if settings.Recorder == "postgres":
            print("실행")
            init_db()

        globals.init_manager()

        app = QApplication(sys.argv)
        window = VantageUI()
        window.show()
        sys.exit(app.exec())