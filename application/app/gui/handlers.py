 
import os
import sys
import threading

from multiprocessing import Process

import app.core.globals as g_vars
from app.services.train.train import TrainMode
import app.services.inference.inferece as inferece
import app.services.inference.inferece_json as inferece_json
from app.gui.plots.plot import plot_main

import app.services.recorders.userMouse as useMouse
from app.utilites.save_confing import update_parameters

from PyQt6.QtWidgets import QSpacerItem, QSizePolicy, QMessageBox, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer

class UIHandler:
    def __init__(self, stop_event, parent=None):
        # 1. parent(VantageUI)를 받아야 setup_tray에서 self.parent 사용 가능
        self.parent = parent 
        self.stop_move_event = stop_event
        # exit_application에서 self.ev를 쓰고 계시므로 통일해줍니다.
        self.ev = stop_event 
        self.tray = None

    def _ask_confirm(self, title, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        # --- 가로 크기 강제 확장 코드 ---
        # 최소 500px의 너비를 확보합니다.
        spacer = QSpacerItem(500, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
        layout = msg_box.layout()
        layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())
        # ----------------------------

        return msg_box.exec() == QMessageBox.StandardButton.Yes

    def start_record(self, isUser, record=False):
        if self._ask_confirm("확인", f"유저 마우스 기록={record} 시작하시겠습니까?"):
            self.stop_move_event.clear()            
            thread = threading.Thread(
                target=useMouse.record_mouse_path, 
                kwargs={"record": record, "isUser": isUser, "stop_event": self.stop_move_event, "log_queue": g_vars.LOG_QUEUE},
                daemon=True
            )
            thread.start()
            g_vars.LOG_QUEUE.put("System: Recording thread started.")


    def start_train(self):
        if self._ask_confirm("확인", "학습을 시작하시겠습니까?"):
            self.stop_move_event.clear()
            
            trainer = TrainMode(
                stop_event=self.stop_move_event, 
                log_queue=g_vars.LOG_QUEUE
            )

            threading.Thread(
                target=trainer.main,
                daemon=True
            ).start()
            g_vars.LOG_QUEUE.put("System: Training thread started.")

    def start_inference(self):
        if self._ask_confirm("확인", "탐지를 시작하시겠습니까?"):
            self.stop_move_event.clear()
            threading.Thread(
                target=inferece.main,
                kwargs={"stop_event": self.stop_move_event, "log_queue": g_vars.LOG_QUEUE},
                daemon=True
            ).start()
            g_vars.LOG_QUEUE.put("System: Inference thread started.")

    def start_inference_json(self):
        if self._ask_confirm("확인", "탐지를 시작하시겠습니까?"):
            self.stop_move_event.clear()
            threading.Thread(
                target=inferece_json.main,
                kwargs={"stop_event": self.stop_move_event, "log_queue": g_vars.LOG_QUEUE},
                daemon=True
            ).start()
            g_vars.LOG_QUEUE.put("System: Inference thread started.")

    def make_plot(self, user=False):
        # 1440px UI에서 버튼 연동을 위해 추가
        try:
            from tkinter import filedialog     
            import json

            points:list[dict]

            file_pahh = filedialog.askopenfilename(title="Json 파일을 선택해 주세요", filetypes=(("json 파일", "*.json"), ("모든 파일", "*.*")))
            if not os.path.exists(file_pahh):
                return [] 

            try:
                with open(file_pahh, "r", encoding="utf-8") as f:
                    data = json.load(f)
            
                points = data
            except Exception as e:
                print(e)
                points = []

            Process(
                target=plot_main, 
                kwargs={"points": points, "log_queue": g_vars.LOG_QUEUE},
                daemon=True
            ).start()
        except Exception as e:
            g_vars.LOG_QUEUE.put(f"Plot Error: {e}")

    def setup_tray(self):
        if not self.parent:
            return

        self.tray = QSystemTrayIcon(self.parent)
        self.tray.setIcon(self.parent.style().standardIcon(
            self.parent.style().StandardPixmap.SP_ComputerIcon
        ))

        menu = QMenu()
        
        # [수정] "창 열기" 시 아이콘을 숨기는 함수 연결
        show_action = QAction("창 열기", self.parent)
        show_action.triggered.connect(self.restore_from_tray) # 커스텀 함수로 변경
        
        exit_action = QAction("완전 종료", self.parent)
        exit_action.triggered.connect(self.exit_application)
        
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        
        self.tray.setContextMenu(menu)
        
        # [수정] 더블클릭 시에도 아이콘을 숨기는 함수 연결
        self.tray.activated.connect(self.on_tray_activated)
        
        # 처음 실행 시에는 창이 떠 있을 것이므로 트레이 아이콘은 숨김 상태로 시작
        self.tray.hide()

    def restore_from_tray(self):
        """창을 다시 표시하고 트레이 아이콘을 숨깁니다."""
        self.parent.showNormal()
        self.parent.activateWindow() # 창을 맨 앞으로
        self.tray.hide() # 아이콘 사라짐

    def on_tray_activated(self, reason):
        """
        하나로 통합된 활성화 함수
        """
        # Trigger(원클릭)와 DoubleClick(더블클릭) 둘 다 허용
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, 
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            QTimer.singleShot(100, self.restore_from_tray)

    def exit_application(self):
        # 종료 전 트레이 아이콘 숨기기 (윈도우 잔상 방지)
        if self.tray:
            self.tray.hide()
            
        self.ev.set()
        g_vars.LOG_QUEUE.put("System: Application shutting down...")
        
        # PyQt6 애플리케이션 종료 처리
        from PyQt6.QtWidgets import QApplication
        QApplication.quit() 
        
        import os
        os._exit(0)

    def update_parameters(self, data_dict):
        update_parameters(data_dict)