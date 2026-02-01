# main.py
from app.gui.main_window import VantageUI
from app.db.session import init_db
from app.core.settings import settings
import app.core.globals as globals
import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

if __name__ == "__main__":
    if settings.Recorder == "postgres":
        print("실행")
        init_db()

    globals.init_manager()

    app = QApplication(sys.argv)
    window = VantageUI()
    window.show()
    sys.exit(app.exec())