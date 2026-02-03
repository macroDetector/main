import os
import sys
import keyboard
from app.utilites.resource_monitoring import ResourceMonitor
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QLineEdit, QTextEdit, QScrollArea, QComboBox, 
                             QSlider, QGridLayout, QSystemTrayIcon)
from PyQt6.QtCore import Qt, QTimer
from multiprocessing import Event

import app.core.globals as g_vars
from app.gui import UIHandler

class VantageUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.font_family = "Segoe UI"
        self.font_size = 12
        self.current_theme = "dark"

        self.stop_move_event = Event()

        self.handler = UIHandler(self.stop_move_event, parent=self)        
        self.monitor = ResourceMonitor()

        self.themes = {
            "dark": {
                "bg": "#121417", "sidebar": "#0B0C0E", "card": "#1C1F23",
                "accent": "#AF966E",
                "text": "#DCDCDC", "text_dim": "#888E96",
                "btn": "#25282D", "input_bg": "#0B0C0E", "terminal": "#08090A", "border": "#2D3137"
            },
            "light": {
                "bg": "#F0F2F5", "sidebar": "#E1E4ED", "card": "#FFFFFF",
                "accent": "#007BFF", "text": "#1A1C26", "text_dim": "#666666",
                "btn": "#E9ECEF", "input_bg": "#FFFFFF", "terminal": "#FDFDFD", "border": "#D1D5DB"
            }
        }

        self.init_ui()
        self.handler.setup_tray()
        self.apply_theme()

        # 타이머 설정
        self.res_timer = QTimer(self)
        self.res_timer.timeout.connect(self.update_resource_labels)
        self.res_timer.start(500)

        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.process_logs)
        self.log_timer.start(50)

        keyboard.add_hotkey('ctrl+shift+q', self.trigger_stop_event)
        self.setMinimumSize(1440, 1010)
        
    # --- [NEW] 공통 스타일 버튼 생성 함수 ---
    def create_styled_button(self, text, cmd, h=45, w=None, obj_name=None, fixed_font_size=None):
        """
        fixed_font_size 파라미터를 추가하여 특정 버튼만 폰트 크기를 고정할 수 있게 합니다.
        """
        btn = QPushButton(text)
        if obj_name:
            btn.setObjectName(obj_name)
        
        btn.setFixedHeight(h)
        if w:
            btn.setFixedWidth(w)
            
        # --- [추가] 폰트 사이즈 고정 로직 ---
        if fixed_font_size:
            btn.setStyleSheet(f"font-size: {fixed_font_size}px !important;")
        # ----------------------------------

        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(cmd)
        return btn

    def hide_to_tray(self):
        """창을 숨기고 트레이 아이콘을 보이게 합니다."""
        self.hide()
        if hasattr(self.handler, 'tray') and self.handler.tray:
            self.handler.tray.show()  # 아이콘 나타남
            self.handler.tray.showMessage(
                "Vantage Controller",
                "백그라운드에서 실행 중입니다.",
                QSystemTrayIcon.MessageIcon.Information,
                1000
            )
            
    def init_ui(self):
        self.setWindowTitle("Controller | Intelligence Control Center")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- [COL 1] 사이드바 ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(280)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(0, 30, 0, 0)

        # 리소스 카드
        res_card = QFrame(); res_card.setObjectName("Card")
        res_lay = QVBoxLayout(res_card)
        res_title = QLabel("APP RESOURCE USAGE")
        res_title.setStyleSheet("font-size: 12px !important;")        
        res_lay.addWidget(res_title)
        
        self.app_cpu_label = QLabel("App CPU: 0.0%")
        self.app_ram_label = QLabel("App RAM: 0.0 MB")
        self.app_gpu_label = QLabel("App GPU: 0.0 MB")

        for lbl in [self.app_cpu_label, self.app_ram_label, self.app_gpu_label]:
            lbl.setObjectName("ResourceLabel")
            lbl.setStyleSheet("font-size: 12px !important; background: transparent;") 
            res_lay.addWidget(lbl)
        side_layout.addWidget(res_card)

        # 설정 카드
        conf_group = QFrame(); conf_group.setObjectName("Card")
        conf_lay = QVBoxLayout(conf_group)
        conf_lay.addWidget(QLabel("INTERFACE SETTINGS"))
        
        # [함수 적용] 테마 스위치 버튼
        self.theme_btn = self.create_styled_button("SWITCH THEME", self.toggle_theme, h=35)
        conf_lay.addWidget(self.theme_btn)

        self.font_combo = QComboBox()
        self.font_combo.setStyleSheet("font-size: 12px !important;")
        self.font_combo.addItems(["Segoe UI", "Malgun Gothic", "Arial", "Consolas"])
        self.font_combo.currentTextChanged.connect(self.update_font)
        conf_lay.addWidget(self.font_combo)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(8, 24); self.size_slider.setValue(12)
        self.size_slider.valueChanged.connect(self.update_font)
        conf_lay.addWidget(self.size_slider)
        side_layout.addWidget(conf_group)

        # --- [추가] 트레이 모드 버튼 ---
        self.tray_mode_btn = self.create_styled_button(
            "HIDE TO TRAY", 
            self.hide_to_tray, 
            h=35, 
            obj_name="TrayBtn"
        )
        side_layout.addWidget(self.tray_mode_btn)
        # ----------------------------

        side_layout.addStretch()
        
        self.ver_label = QLabel("ver 0.0.2\nDev by qqqqaqaqaqq")
        self.ver_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ver_label.setStyleSheet("font-size: 10px !important; color: var(--text_dim); margin-bottom: 10px;")
        side_layout.addWidget(self.ver_label)
        
        self.main_layout.addWidget(self.sidebar)

        # --- [COL 2] 컨트롤 센터 ---
        self.control_panel = QFrame()
        self.control_panel.setObjectName("ControlPanel")
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(30, 40, 30, 40)

        header = QLabel("CONTROL CENTER")
        header.setObjectName("MainHeader")
        header.setStyleSheet("font-family: 'Impact'; font-size: 32px;") 
        control_layout.addWidget(header)

        hotkey_lbl = QLabel("● HOTKEY: CTRL + SHIFT + Q TO STOP")
        hotkey_lbl.setObjectName("HotKeyLabel")
        control_layout.addWidget(hotkey_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self.scroll_layout = QVBoxLayout(scroll_content)
        
        # 섹션들 추가
        self.scroll_layout.addWidget(self.create_section("🎥 MOUSE CAPTURE", [
            ("Start New Mouse Recording", lambda: self.handler.start_record(isUser=True, record=True))
        ]))
        
        self.scroll_layout.addWidget(self.create_combined_settings_card())
        
        # [함수 적용] 시각 분석 카드
        plot_card = QFrame(); plot_card.setObjectName("Card")
        p_lay = QVBoxLayout(plot_card)
        p_lay.addWidget(QLabel("📊 VISUAL ANALYSIS"))
        u_plot_btn = self.create_styled_button("PLOT USER PATH", lambda: self.handler.make_plot(user=True), h=50)
        p_lay.addWidget(u_plot_btn)
        self.scroll_layout.addWidget(plot_card)

        self.scroll_layout.addWidget(self.create_section("🧠 AI ENGINE", [
            ("Run Model Training", self.handler.start_train),
            ("Start Real-time Inference", self.handler.start_inference)
        ]))

        
        self.scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        control_layout.addWidget(scroll)
        
        self.main_layout.addWidget(self.control_panel, stretch=1)

        # --- [COL 3] 터미널 ---
        self.terminal_area = QFrame()
        term_layout = QVBoxLayout(self.terminal_area)
        term_layout.setContentsMargins(20, 40, 20, 20)

        term_header = QHBoxLayout()
        term_header.addWidget(QLabel("SYSTEM TERMINAL LOGS"))
        
        # [함수 적용] CLEAR 버튼 (가로 80 고정)
        self.clear_btn = self.create_styled_button("CLEAR", self.clear_logs, h=30, w=80, fixed_font_size=11)
        term_header.addWidget(self.clear_btn)
        
        term_layout.addLayout(term_header)
        self.macro_text = QTextEdit(); self.macro_text.setReadOnly(True)
        term_layout.addWidget(self.macro_text)
        
        self.main_layout.addWidget(self.terminal_area, stretch=1)

    def create_section(self, title, buttons):
        card = QFrame(); card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.addWidget(QLabel(title))
        for text, cmd in buttons:
            # [함수 적용]
            btn = self.create_styled_button(text, cmd, h=45)
            lay.addWidget(btn)
        return card


    def create_combined_settings_card(self):
        card = QFrame(); card.setObjectName("Card")
        lay = QVBoxLayout(card)
        title = QLabel("⚙️ SYSTEM & MODEL PARAMETERS")
        title.setStyleSheet("font-weight: bold; color: var(--accent);")
        lay.addWidget(title)
        
        grid = QGridLayout()
        self.inputs['SEQ_LEN'] = self.add_grid_input(grid, "SEQ_LEN", str(g_vars.SEQ_LEN), 0, 0)
        self.inputs['STRIDE'] = self.add_grid_input(grid, "STRIDE", str(g_vars.STRIDE), 0, 1)
        self.inputs['HIDDEN'] = self.add_grid_input(grid, "HIDDEN", str(g_vars.d_model), 1, 0)
        self.inputs['LAYERS'] = self.add_grid_input(grid, "LAYERS", str(g_vars.num_layers), 1, 1)
        self.inputs['LR'] = self.add_grid_input(grid, "LR", str(g_vars.lr), 2, 0)
        self.inputs['THRES'] = self.add_grid_input(grid, "THRESHOLD", str(g_vars.threshold), 2, 1)
        self.inputs['TOLE'] = self.add_grid_input(grid, "TOLERANCE", str(g_vars.tolerance), 3, 0)
        lay.addLayout(grid)

        # [함수 적용] SAVE 버튼 (높이 50, 별도 스타일 적용 위해 ObjectName 부여)
        self.apply_all_btn = self.create_styled_button(
            "SAVE & APPLY PARAMETERS", 
            self.apply_params, 
            h=50, 
            obj_name="ApplyBtn"
        )
        lay.addWidget(self.apply_all_btn)
        return card

    def add_grid_input(self, layout, label, default, r, c):
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel(label))
        edit = QLineEdit(default); edit.setFixedHeight(35)
        vbox.addWidget(edit); layout.addLayout(vbox, r, c)
        return edit

    def apply_theme(self):
        c = self.themes[self.current_theme]
        base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        css_path = os.path.join(base_path, "style.css")

        try:
            with open(css_path, "r", encoding="utf-8") as f:
                style = f.read()
            
            replacements = {
                "var(--bg)": c['bg'], "var(--text)": c['text'], "var(--sidebar)": c['sidebar'],
                "var(--card)": c['card'], "var(--border)": c['border'], "var(--btn)": c['btn'],
                "var(--accent)": c['accent'], "var(--input_bg)": c['input_bg'], "var(--terminal)": c['terminal'],
                "var(--text_dim)": c['text_dim'], "var(--font_family)": self.font_family, 
                "var(--font_size)": str(self.font_size)
            }
            for p, v in replacements.items(): 
                style = style.replace(p, v)
            
            self.setStyleSheet(style)
            self.control_panel.setStyleSheet(f"#ControlPanel {{ background-color: {c['bg']}; border: none; }}")
            self.terminal_area.setStyleSheet(f"#TerminalArea {{ background-color: {c['bg']}; border: none; }}")
            self.sidebar.setStyleSheet(f"#Sidebar {{ background-color: {c['sidebar']}; border-right: 1px solid {c['border']}; }}")
            
        except Exception as e:
            print(f"Theme Error: {e}")

    def trigger_stop_event(self):
        self.stop_move_event.set()
        g_vars.LOG_QUEUE.put("🛑 STOP SIGNAL RECEIVED (CTRL+SHIFT+Q)")

    def update_resource_labels(self):
        try:
            # 1. 기존 리소스 모니터링
            stats = self.monitor.get_stats()
            self.app_cpu_label.setText(f"App CPU: {stats['cpu']}")
            self.app_ram_label.setText(f"App RAM: {stats['ram']}")
            self.app_gpu_label.setText(f"App GPU: {stats['gpu']}")

            # 2. [추가] 외부에서 변경된 g_vars 값을 UI에 실시간 동기화
            # 현재 입력창이 포커스(커서)가 없을 때만 업데이트 (사용자 입력 방해 방지)
            sync_map = {
                'SEQ_LEN': str(g_vars.SEQ_LEN),
                'STRIDE': str(g_vars.STRIDE),
                'HIDDEN': str(g_vars.d_model),
                'LAYERS': str(g_vars.num_layers),
                'LR': str(g_vars.lr),
                'THRES': str(g_vars.threshold),
                'TOLE': str(g_vars.tolerance)
            }

            for key, current_val in sync_map.items():
                edit_widget = self.inputs.get(key)
                if edit_widget and not edit_widget.hasFocus(): # 사용자가 입력 중이 아닐 때만
                    if edit_widget.text() != current_val:
                        edit_widget.setText(current_val)
                        
        except Exception as e:
            print(f"Sync Error: {e}")
            
    def update_font(self):
        self.font_family = self.font_combo.currentText()
        self.font_size = self.size_slider.value()
        self.apply_theme()

    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()

    def process_logs(self):
        while not g_vars.LOG_QUEUE.empty():
            self.macro_text.append(f"> {g_vars.LOG_QUEUE.get()}")

    def clear_logs(self):
        self.macro_text.clear()
        g_vars.LOG_QUEUE.put("🧹 Terminal logs cleared.")

    def apply_params(self):
        data_to_save = {key: edit.text() for key, edit in self.inputs.items()}
        success = self.handler.update_parameters(data_to_save)
        if success: pass

    def closeEvent(self, event):
        self.stop_move_event.set()
        # 종료할 때는 트레이 아이콘도 깔끔하게 제거
        if hasattr(self.handler, 'tray'):
            self.handler.tray.hide()
        os._exit(0)