import sys
import os

def get_resource_path(relative_path):
    """ 개발 환경과 PyInstaller 환경 모두에서 작동하는 경로 변환 함수 """
    if getattr(sys, 'frozen', False):
        # 빌드 후 (.exe 실행 시)
        base_path = sys._MEIPASS
    else:
        # 개발 환경 (python main.py 실행 시)
        # 현재 파일(main_window.py) 위치: .../application/app/gui/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 프로젝트 루트(application/) 폴더로 가기 위해 상위로 2번 이동
        # gui -> app (1번), app -> application (2번)
        base_path = os.path.dirname(os.path.dirname(current_dir))

    # 경로를 합친 후 표준 형식으로 정리
    return os.path.normpath(os.path.join(base_path, relative_path))