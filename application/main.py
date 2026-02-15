import sys
import multiprocessing 

def initialize_system():
    # 1. 가장 먼저 보이는 메시지
    print("🚀  프로그램 실행 중... 잠시만 기다려 주세요.")

    # 2. 가장 무거운 torch 로딩 시각화
    print("📦 라이브러리 로드 중 (PyTorch)...", end="\r")
    import torch # import를 여기서 해도 됩니다 (함수 내부 import)
    print("📦  라이브러리 로드 완료 (PyTorch)   ")

    print("⚙️  시스템 환경 설정 중...", end="\r")
    import app.core.globals as g_vars
    from app.core.settings import settings
    print("⚙️  시스템 환경 설정 완료         ")

    return g_vars, settings

if __name__ == "__main__":
    multiprocessing.freeze_support()
    

    import platform
    IS_WINDOWS = platform.system() == "Windows"
    
    if IS_WINDOWS:
        from app.utilites.yncheck import yncheck
        g_vars, settings = initialize_system()

        print("Welcome")
        user_input = input("inference Mode? (y/n): ").lower()

        if not yncheck(user_input):
            sys.exit()

        inference_Mode = user_input == 'y'
        
        if inference_Mode:
            from app.cli.windowmode import windowmode
            windowmode()

        else:
            from app.gui.main_window import VantageUI
            from PyQt6.QtWidgets import QApplication
            import app.core.globals as g_vars
        
            g_vars.init_manager()

            app = QApplication(sys.argv)
            window = VantageUI()
            window.show()
            sys.exit(app.exec())
    