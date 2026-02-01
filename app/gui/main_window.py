import os
import sys
import keyboard

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QLineEdit, QTextEdit, QScrollArea, QComboBox, 
                             QSlider, QGridLayout, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
from multiprocessing import Event, Queue

# [Global/Core 연동 섹션]
try:
    import app.core.globals as globals
    from app.gui.handlers import UIHandler
    print(f"✅ Real Handlers & Globals Loaded")
except ImportError:
    # 모듈이 없을 경우를 대비한 Mock 데이터 (테스트/독립 실행용)
    class Mock: pass
    globals = Mock()
    globals.SEQ_LEN = 100; globals.STRIDE = 10; globals.d_model = 128; 
    globals.num_layers = 2; globals.lr = 0.001
    globals.LOG_QUEUE = Queue()
    class UIHandler:
        def __init__(self, ev): self.ev = ev
        def start_record(self, **kwargs): globals.LOG_QUEUE.put("Recording Started...")
        def start_train(self): globals.LOG_QUEUE.put("Training Started...")
        def start_inference(self): globals.LOG_QUEUE.put("Inference Started...")
        def make_plot(self, user=False): globals.LOG_QUEUE.put(f"📊 Plotting {'User' if user else 'Bot'} path data...")

class VantageUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.font_family = "Segoe UI"
        self.font_size = 10
        self.current_theme = "dark"
        
        # 중단 이벤트 및 핸들러 설정
        self.stop_move_event = Event()
        self.handler = UIHandler(self.stop_move_event)
        
        # 단축키 설정 (시스템 전역)
        keyboard.add_hotkey('ctrl+shift+q', self.trigger_stop_event)

        # UI 최소 사이즈 설정
        self.setMinimumSize(1440, 1024)

        self.themes = {
            "dark": {
                "bg": "#0F111A", "sidebar": "#08090D", "card": "#1A1C26",
                "accent": "#00E5FF", "text": "#FFFFFF", "text_dim": "#9499C3",
                "btn": "#2D303E", "input_bg": "#0F111A", "terminal": "#0A0B10", "border": "#252A34"
            },
            "light": {
                "bg": "#F0F2F5", "sidebar": "#E1E4ED", "card": "#FFFFFF",
                "accent": "#007BFF", "text": "#1A1C26", "text_dim": "#666666",
                "btn": "#E9ECEF", "input_bg": "#FFFFFF", "terminal": "#FDFDFD", "border": "#D1D5DB"
            }
        }

        self.init_ui()
        self.apply_theme()

        # 로그 업데이트 타이머 (GUI 스레드 점유 방지)
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.process_logs)
        self.log_timer.start(50)

    def trigger_stop_event(self):
        self.stop_move_event.set()
        globals.LOG_QUEUE.put("🛑 STOP SIGNAL RECEIVED (CTRL+SHIFT+Q)")

    def init_ui(self):
        self.setWindowTitle("Controller | Intelligence Control Center")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- [COL 1] 사이드바 (설정) ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(280)
        side_layout = QVBoxLayout(self.sidebar)
        
        self.logo_label = QLabel("VANTAGE")
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_layout.addWidget(self.logo_label)

        conf_group = QFrame(); conf_group.setObjectName("Card")
        conf_lay = QVBoxLayout(conf_group)
        conf_lay.addWidget(QLabel("INTERFACE SETTINGS"))
        
        self.theme_btn = QPushButton("SWITCH THEME")
        self.theme_btn.setFixedHeight(40)
        self.theme_btn.clicked.connect(self.toggle_theme)
        conf_lay.addWidget(self.theme_btn)

        conf_lay.addSpacing(10)
        conf_lay.addWidget(QLabel("Font Family:"))
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Segoe UI", "Malgun Gothic", "Arial", "Consolas"])
        self.font_combo.currentTextChanged.connect(self.update_font)
        conf_lay.addWidget(self.font_combo)

        conf_lay.addWidget(QLabel("Font Size:"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(8, 20)
        self.size_slider.setValue(10)
        self.size_slider.valueChanged.connect(self.update_font)
        conf_lay.addWidget(self.size_slider)
        
        side_layout.addWidget(conf_group)
        side_layout.addStretch()
        self.main_layout.addWidget(self.sidebar)

        # --- [COL 2] 컨트롤 센터 (600px) ---
        self.control_panel = QFrame()
        self.control_panel.setFixedWidth(600)
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(30, 40, 30, 40)
        
        self.header_label = QLabel("CONTROL CENTER")
        self.header_label.setStyleSheet("font-family: 'Impact'; font-size: 32px;")
        control_layout.addWidget(self.header_label)

        self.shortcut_label = QLabel("  ● HOTKEY: CTRL + SHIFT + Q TO STOP")
        self.shortcut_label.setStyleSheet("color: #FF4B4B; font-weight: bold; margin-bottom: 20px;")
        control_layout.addWidget(self.shortcut_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setSpacing(20)

        # 1. MOUSE CAPTURE 섹션
        self.scroll_layout.addWidget(self.create_section("🎥 MOUSE CAPTURE", [
            ("Start New Mouse Recording", lambda: self.handler.start_record(isUser=True, record=True))
        ]))
        
        # 2. SYSTEM PARAMETERS (Grid Layout)
        self.scroll_layout.addWidget(self.create_combined_settings_card())
        
        # 3. VISUAL ANALYSIS (Bot 제외, User Path 버튼만 유지)
        plot_card = QFrame(); plot_card.setObjectName("Card")
        p_lay = QVBoxLayout(plot_card)
        p_lay.addWidget(QLabel("📊 VISUAL ANALYSIS"))
        
        self.u_plot_btn = QPushButton("PLOT USER PATH")
        self.u_plot_btn.setFixedHeight(50)
        self.u_plot_btn.clicked.connect(lambda: self.handler.make_plot(user=True))
        
        p_lay.addWidget(self.u_plot_btn) # 버튼 직접 추가하여 누락 방지
        self.scroll_layout.addWidget(plot_card)

        # 4. AI ENGINE 섹션
        self.scroll_layout.addWidget(self.create_section("🧠 AI ENGINE", [
            ("Run Model Training", self.handler.start_train),
            ("Start Real-time Inference", self.handler.start_inference)
        ]))

        self.scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        control_layout.addWidget(scroll)
        self.main_layout.addWidget(self.control_panel)

        # --- [COL 3] 터미널 (우측 나머지) ---
        self.terminal_area = QFrame()
        term_layout = QVBoxLayout(self.terminal_area)
        term_layout.setContentsMargins(20, 40, 20, 20)
        
        term_layout.addWidget(QLabel("SYSTEM TERMINAL LOGS"))
        self.macro_text = QTextEdit()
        self.macro_text.setReadOnly(True)
        term_layout.addWidget(self.macro_text)
        self.main_layout.addWidget(self.terminal_area, stretch=1)

    # [Helper Methods]
    def create_section(self, title, buttons):
        card = QFrame(); card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.addWidget(QLabel(title))
        for text, cmd in buttons:
            btn = QPushButton(text); btn.setFixedHeight(45); btn.clicked.connect(cmd)
            lay.addWidget(btn)
        return card

    def create_combined_settings_card(self):
        card = QFrame(); card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.addWidget(QLabel("⚙️ SYSTEM & MODEL PARAMETERS"))
        
        grid_lay = QGridLayout()
        self.inputs['SEQ_LEN'] = self.add_grid_input(grid_lay, "SEQ_LEN", str(globals.SEQ_LEN), 0, 0)
        self.inputs['STRIDE'] = self.add_grid_input(grid_lay, "STRIDE", str(globals.STRIDE), 0, 1)
        self.inputs['HIDDEN'] = self.add_grid_input(grid_lay, "HIDDEN", str(globals.d_model), 1, 0)
        self.inputs['LAYERS'] = self.add_grid_input(grid_lay, "LAYERS", str(globals.num_layers), 1, 1)
        self.inputs['LR'] = self.add_grid_input(grid_lay, "LR", str(globals.lr), 2, 0)
        lay.addLayout(grid_lay)

        lay.addSpacing(10)
        self.apply_all_btn = QPushButton("SAVE & APPLY PARAMETERS")
        self.apply_all_btn.setFixedHeight(50)
        self.apply_all_btn.setObjectName("ApplyBtn")
        self.apply_all_btn.clicked.connect(self.apply_params)
        lay.addWidget(self.apply_all_btn)
        return card

    def add_grid_input(self, layout, label, default, r, c):
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel(label))
        edit = QLineEdit(default); edit.setFixedHeight(35)
        vbox.addWidget(edit)
        layout.addLayout(vbox, r, c)
        return edit

    def apply_params(self):
        try:
            s_len = int(self.inputs['SEQ_LEN'].text().strip())
            stride = int(self.inputs['STRIDE'].text().strip())
            hidden = int(self.inputs['HIDDEN'].text().strip())
            layers = int(self.inputs['LAYERS'].text().strip())
            lr = float(self.inputs['LR'].text().strip())

            globals.SEQ_LEN = s_len
            globals.STRIDE = stride
            globals.d_model = hidden
            globals.num_layers = layers
            globals.lr = lr

            globals.LOG_QUEUE.put(f"[INFO] Parameters Applied: SEQ={s_len}, HIDDEN={hidden}")

            # .env 업데이트 로직
            env_path = ".env"
            env_dict = {}
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "=" in line:
                            k, v = line.strip().split("=", 1)
                            env_dict[k] = v
            
            env_dict.update({
                "SEQ_LEN": str(s_len), "STRIDE": str(stride),
                "d_model": str(hidden), "num_layers": str(layers), "lr": str(lr)
            })
            
            with open(env_path, "w", encoding="utf-8") as f:
                for k, v in env_dict.items(): f.write(f"{k}={v}\n")
            globals.LOG_QUEUE.put(f"[INFO] .env file updated.")

        except Exception as e:
            globals.LOG_QUEUE.put(f"[ERROR] Update failed: {str(e)}")

    def update_font(self):
        self.font_family = self.font_combo.currentText()
        self.font_size = self.size_slider.value()
        self.apply_theme()

    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()

    def apply_theme(self):
        c = self.themes[self.current_theme]
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {c['bg']}; color: {c['text']}; font-family: '{self.font_family}'; font-size: {self.font_size}px; }}
            QFrame#Sidebar {{ background-color: {c['sidebar']}; border-right: 1px solid {c['border']}; }}
            QFrame#Card {{ background-color: {c['card']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 15px; }}
            QPushButton {{ background-color: {c['btn']}; border-radius: 6px; padding: 5px; font-weight: bold; border: 1px solid {c['border']}; }}
            QPushButton:hover {{ background-color: {c['accent']}; color: #000; }}
            QPushButton#ApplyBtn {{ background-color: {c['accent']}; color: #000; font-size: 13px; }}
            QLineEdit, QComboBox {{ background-color: {c['input_bg']}; color: {c['accent']}; border: 1px solid {c['border']}; border-radius: 4px; padding: 4px; }}
            QTextEdit {{ background-color: {c['terminal']}; color: {c['accent']}; font-family: 'Consolas'; border-radius: 8px; padding: 15px; border: 1px solid {c['border']}; }}
            QLabel {{ font-weight: bold; background: transparent; }}
        """)
        self.logo_label.setStyleSheet(f"color: {c['accent']}; font-family: 'Impact'; font-size: 36px; margin: 40px 0;")

    def process_logs(self):
        while not globals.LOG_QUEUE.empty():
            log_msg = globals.LOG_QUEUE.get()
            self.macro_text.append(f"> {log_msg}")

    def closeEvent(self, event):
        """윈도우 종료 시 호출되는 완전 종료 로직"""
        print("Safely shutting down...")
        self.stop_move_event.set()
        
        if hasattr(self, 'log_timer'):
            self.log_timer.stop()
            
        try:
            keyboard.unhook_all()
        except Exception:
            pass

        # 멀티프로세싱 큐 비우기 (Hang 방지)
        while not globals.LOG_QUEUE.empty():
            try:
                globals.LOG_QUEUE.get_nowait()
            except:
                break

        print("Bye!")
        event.accept()
        # 모든 스레드와 자원을 강제 해제하고 종료
        os._exit(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VantageUI()
    gui.show()
    sys.exit(app.exec())