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
from app.utilites.get_resource_path import get_resource_path

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
        self.setMinimumSize(1440, 900)
        
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
        
        self.ver_label = QLabel("ver 0.0.3\nDev by qqqqaqaqaqq")
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
            ("Start New Mouse Recording feat User", lambda: self.handler.start_record(isUser=True, record=True)),
            ("Start New Mouse Recording feat Move_Data", lambda: self.handler.start_record(isUser=False, record=True))            
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
            ("Start Real-time Inference", self.handler.start_inference),
            ("Json Data Inference", self.handler.start_inference_json)            
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


    # --- [수정] 툴팁(Hint) 기능을 포함한 입력창 생성 함수 ---
    def add_grid_input(self, layout, label, default, r, c, hint=None):
        vbox = QVBoxLayout()
        lbl_widget = QLabel(label)
        vbox.addWidget(lbl_widget)
        
        edit = QLineEdit(default)
        edit.setFixedHeight(35)
        
        # 힌트(툴팁)가 있으면 추가 (HTML 태그로 꾸미기 가능)
        if hint:
            edit.setToolTip(hint)
            lbl_widget.setToolTip(hint) # 라벨에 올려도 보이게 처리
            
        vbox.addWidget(edit)
        layout.addLayout(vbox, r, c)
        return edit

    # --- [수정] 설정 카드 생성 부분 (상세 힌트 추가) ---
    def create_combined_settings_card(self):
        card = QFrame(); card.setObjectName("Card")
        lay = QVBoxLayout(card)
        
        title = QLabel("⚙️ SYSTEM & MODEL PARAMETERS")
        title.setStyleSheet("font-weight: bold; color: var(--accent); font-size: 16px; margin-bottom: 10px;")
        lay.addWidget(title)
        
        grid = QGridLayout()
        grid.setSpacing(12) 

        # --- 그룹 1: RECORDER & DATA ---
        grid.addWidget(QLabel("📍 DATA RECORDER"), 0, 0, 1, 3)
        self.inputs['SEQ_LEN'] = self.add_grid_input(grid, "SEQ_LEN", str(g_vars.SEQ_LEN), 1, 0, 
            "<b>Window Size</b>: 모델이 한 번에 볼 과거 데이터의 길이입니다.")
        self.inputs['STRIDE'] = self.add_grid_input(grid, "STRIDE", str(g_vars.STRIDE), 1, 1, 
            "데이터를 슬라이딩할 간격입니다. 낮을수록 데이터 양이 많아집니다.")
        self.inputs['TOLE'] = self.add_grid_input(grid, "TOLERANCE", str(g_vars.tolerance), 1, 2, 
            "데이터 수집 시 변화를 감지할 최소 허용 오차입니다.")

        # --- 그룹 2: MODEL ARCHITECTURE ---
        grid.addWidget(QLabel("🧠 ARCHITECTURE"), 2, 0, 1, 3)
        self.inputs['D_MODEL'] = self.add_grid_input(grid, "D_MODEL", str(g_vars.d_model), 3, 0, 
            "모델 내부의 임베딩 차원 크기입니다.")
        self.inputs['N_HEAD'] = self.add_grid_input(grid, "N_HEAD", str(g_vars.n_head), 3, 1, 
            "Attention Head의 개수입니다. (D_MODEL의 약수여야 함)")
        self.inputs['LAYERS'] = self.add_grid_input(grid, "LAYERS", str(g_vars.num_layers), 3, 2, 
            "Transformer 레이어의 깊이입니다.")
        self.inputs['FEED'] = self.add_grid_input(grid, "FEED_FWD", str(g_vars.dim_feedforward), 4, 0, 
            "FFN 레이어의 내부 차원 크기입니다.")
        self.inputs['DROP'] = self.add_grid_input(grid, "DROP_OUT", str(g_vars.dropout), 4, 1, 
            "<b>Dropout</b>: 과적합 방지를 위해 뉴런을 끄는 비율입니다. (0.1~0.3 권장)")

        # --- 그룹 3: TRAINING ---
        grid.addWidget(QLabel("🚀 TRAINING"), 5, 0, 1, 3)
        self.inputs['LR'] = self.add_grid_input(grid, "LR", str(g_vars.lr), 6, 0, 
            "학습률입니다. 너무 크면 발산하고 너무 작으면 학습이 느립니다.")
        self.inputs['BATCH'] = self.add_grid_input(grid, "BATCH", str(g_vars.batch_size), 6, 1,
            "한번에 얼마만큼 볼지를 결정하는 값 입니다.")
        self.inputs['EPOCH'] = self.add_grid_input(grid, "EPOCH", str(g_vars.epoch), 6, 2,
            "전체를 몇 번 돌릴지 정하는 값 입니다.")
        self.inputs['WEIGHT'] = self.add_grid_input(grid, "WEIGHT_D", str(g_vars.weight_decay), 7, 0, 
            "L2 정규화 강도입니다. 과적합을 방지합니다.")
        self.inputs['PATIENCE'] = self.add_grid_input(grid, "PATIENCE", str(g_vars.patience), 7, 1, 
            "Early Stopping을 위한 대기 횟수입니다.")

        # --- 그룹 4: INFERENCE ---
        grid.addWidget(QLabel("🔍 INFERENCE"), 8, 0, 1, 3)
        self.inputs['THRES'] = self.add_grid_input(grid, "THRESHOLD", str(g_vars.threshold), 9, 0, 
            "<b>Threshold</b>: 이상치 판단 기준값입니다. 에러가 이보다 크면 이상으로 감지합니다.")
        self.inputs['WEIGHT_THRES'] = self.add_grid_input(grid, "WEIGHT_THRES", str(g_vars.weight_threshold), 9, 1, 
            "<b>Weight Threshold</b>: 이상치 판단의 가중치 입니다 낮을 수록 값에 더 민감 해집니다.")
        
        lay.addLayout(grid)

        self.apply_all_btn = self.create_styled_button("SAVE & APPLY PARAMETERS", self.apply_params, h=50, obj_name="ApplyBtn")
        lay.addWidget(self.apply_all_btn)
        return card
        
    def apply_theme(self):
        """
        테마 색상 및 폰트를 적용하고, 
        특히 보이지 않던 툴팁(Hint)의 스타일을 강제로 설정합니다.
        """
        print("css 불러오는중")
        c = self.themes[self.current_theme]
        css_path = get_resource_path(os.path.join("app", "gui", "style.css"))
    
        print(f"최종 경로: {css_path}")
        print(f"파일 존재 여부: {os.path.exists(css_path)}")

        try:
            # 1. 외부 style.css 읽기
            if os.path.exists(css_path):
                with open(css_path, "r", encoding="utf-8") as f:
                    style = f.read()
                    print("CSS 불러오기 성공")
            else:
                style = "" # 파일이 없을 경우 대비

            # 2. 툴팁 스타일 정의 (힌트가 안 보이는 문제 해결 핵심)
            # 배경은 어둡게, 테두리와 글씨는 강조색(accent)으로 설정
            tooltip_style = f"""
            QToolTip {{
                background-color: {c['card']} !important;
                color: {c['accent']} !important;
                border: 1px solid {c['accent']};
                padding: 8px;
                border-radius: 4px;
                font-family: '{self.font_family}';
                font-size: {max(11, self.font_size - 1)}px;
            }}
            """
            
            # 기존 스타일시트에 툴팁 스타일 합치기
            style += tooltip_style

            # 3. 변수 치환 (CSS 변수 대응)
            replacements = {
                "var(--bg)": c['bg'], 
                "var(--text)": c['text'], 
                "var(--sidebar)": c['sidebar'],
                "var(--card)": c['card'], 
                "var(--border)": c['border'], 
                "var(--btn)": c['btn'],
                "var(--accent)": c['accent'], 
                "var(--input_bg)": c['input_bg'], 
                "var(--terminal)": c['terminal'],
                "var(--text_dim)": c['text_dim'], 
                "var(--font_family)": self.font_family, 
                "var(--font_size)": str(self.font_size)
            }
            
            for p, v in replacements.items(): 
                style = style.replace(p, v)
            
            # 4. 앱 전체 및 주요 패널에 적용
            self.setStyleSheet(style)
            
            # 개별 위젯 스타일 강제 재설정 (ID 기반)
            self.control_panel.setStyleSheet(f"#ControlPanel {{ background-color: {c['bg']}; border: none; }}")
            self.terminal_area.setStyleSheet(f"#TerminalArea {{ background-color: {c['bg']}; border: none; }}")
            self.sidebar.setStyleSheet(f"#Sidebar {{ background-color: {c['sidebar']}; border-right: 1px solid {c['border']}; }}")
            
            # 5. 툴팁 폰트 전역 설정 (CSS만으로 부족할 경우 대비)
            from PyQt6.QtGui import QFont
            from PyQt6.QtWidgets import QToolTip
            QToolTip.setFont(QFont(self.font_family, 10))

        except Exception as e:
            print(f"Theme Apply Error: {e}")

    def trigger_stop_event(self):
        self.stop_move_event.set()
        g_vars.LOG_QUEUE.put("🛑 STOP SIGNAL RECEIVED (CTRL+SHIFT+Q)")

    def on_training_finished(self, final_params):
        """학습이 끝났을 때만 UI를 갱신"""
        for key, value in final_params.items():
            if key in self.inputs:
                # 사용자가 입력 중이 아닐 때만 업데이트
                if not self.inputs[key].hasFocus():
                    self.inputs[key].setText(str(value))
        
        # 알림창 하나 띄워주면 더 친절하겠죠?
        print("학습 결과가 UI에 반영되었습니다.")

    def add_grid_input(self, layout, label, default, r, c, hint=None):
        vbox = QVBoxLayout()
        lbl_widget = QLabel(label)
        vbox.addWidget(lbl_widget)
        
        edit = QLineEdit(default)
        edit.setFixedHeight(35)
        
        # 힌트(툴팁)가 있으면 추가 (HTML 태그로 꾸미기 가능)
        if hint:
            edit.setToolTip(hint)
            lbl_widget.setToolTip(hint) # 라벨에 올려도 보이게 처리
            
        vbox.addWidget(edit)
        layout.addLayout(vbox, r, c)
        return edit

    def update_resource_labels(self):
        try:
            # 1. 리소스 모니터링 갱신
            stats = self.monitor.get_stats()
            self.app_cpu_label.setText(f"App CPU: {stats['cpu']}")
            self.app_ram_label.setText(f"App RAM: {stats['ram']}")
            self.app_gpu_label.setText(f"App GPU: {stats['gpu']}")

            # 2. [수정] g_vars -> UI 실시간 동기화 로직 개선
            # 사용자가 입력 창을 클릭 중(Focus)일 때는 자동 덮어쓰기를 중단합니다.
            if g_vars.GLOBAL_CHANGE:
                sync_map = {
                    'SEQ_LEN': str(g_vars.SEQ_LEN),
                    'STRIDE': str(g_vars.STRIDE),
                    'D_MODEL': str(g_vars.d_model),
                    'LAYERS': str(g_vars.num_layers),
                    'LR': str(g_vars.lr),
                    'THRES': str(g_vars.threshold),
                    'TOLE': str(g_vars.tolerance),
                    'N_HEAD':str(g_vars.n_head),
                    'BATCH':str(g_vars.batch_size),
                    "EPOCH" : str(g_vars.epoch),
                    "PATIENCE" : str(g_vars.patience),
                    "WEIGHT" : str(g_vars.weight_decay),
                    "FEED" : str(g_vars.dim_feedforward),
                    "DROP" : str(g_vars.dropout),
                    "WEIGHT_THRES" : str(g_vars.weight_threshold)
                }

                for key, current_gvar_val in sync_map.items():
                    edit_widget = self.inputs.get(key)
                    if edit_widget:
                        # 조건 1: 사용자가 해당 입력창을 수정 중이 아님 (Focus 없음)
                        # 조건 2: 현재 입력창의 텍스트가 실제 g_vars의 값과 다름
                        if not edit_widget.hasFocus() and edit_widget.text() != current_gvar_val:
                            edit_widget.setText(current_gvar_val)

            with g_vars.lock:
                g_vars.GLOBAL_CHANGE = False
            
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