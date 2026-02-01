import threading
from PyQt6.QtWidgets import QMessageBox
from multiprocessing import Process
import app.core.globals as globals
from app.services.train import TrainMode
import app.services.inferece as inferece
from app.services.plot import plot_main
import app.repostitories.DBController as DBController
import app.repostitories.JsonController as JsonController
import app.services.userMouse as useMouse
from PyQt6.QtWidgets import QSpacerItem, QSizePolicy, QGridLayout

class UIHandler:
    def __init__(self, stop_event):
        self.stop_move_event = stop_event

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
                kwargs={"record": record, "isUser": isUser, "stop_event": self.stop_move_event, "log_queue": globals.LOG_QUEUE},
                daemon=True
            )
            thread.start()
            globals.LOG_QUEUE.put("System: Recording thread started.")

    def start_train(self):
        if self._ask_confirm("확인", "학습을 시작하시겠습니까?"):
            self.stop_move_event.clear()
            
            trainer = TrainMode(
                stop_event=self.stop_move_event, 
                log_queue=globals.LOG_QUEUE
            )

            threading.Thread(
                target=trainer.main,
                daemon=True
            ).start()
            globals.LOG_QUEUE.put("System: Training thread started.")

    def start_inference(self):
        if self._ask_confirm("확인", "탐지를 시작하시겠습니까?"):
            self.stop_move_event.clear()
            threading.Thread(
                target=inferece.main,
                kwargs={"stop_event": self.stop_move_event, "log_queue": globals.LOG_QUEUE},
                daemon=True
            ).start()
            globals.LOG_QUEUE.put("System: Inference thread started.")

    def make_plot(self, user=False):
        # 1440px UI에서 버튼 연동을 위해 추가
        try:
            if globals.Recorder == "postgres":
                points = DBController.read(user, log_queue=globals.LOG_QUEUE)
            else:
                points = JsonController.read(user, log_queue=globals.LOG_QUEUE)

            Process(
                target=plot_main, 
                kwargs={"points": points, "log_queue": globals.LOG_QUEUE},
                daemon=True
            ).start()
        except Exception as e:
            globals.LOG_QUEUE.put(f"Plot Error: {e}")

    def clear_db(self):
        if self._ask_confirm("확인", "Mouse DB를 초기화하시겠습니까?"):
            if globals.Recorder == "postgres":
                DBController.point_clear(log_queue=globals.LOG_QUEUE)
                globals.LOG_QUEUE.put("Mouse DB 초기화 완료")
                QMessageBox.information(None, "완료", "초기화가 완료되었습니다.")
            else:
                QMessageBox.warning(None, "경고", "Json 파일을 직접 지워주세요.")