import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np

class RealTimeMonitor:
    def __init__(self, features, threshold):
        self.app = QApplication.instance() or QApplication([])
        # 디자인 개선을 위해 배경색 및 제목 스타일 유지
        self.win = pg.GraphicsLayoutWidget(show=True, title="🚨 Extreme Value Analysis Monitor")
        self.win.resize(1600, 1250) 
        self.win.setBackground('#0A0A0C')

        self.features = features 
        self.num_features = len(self.features)
        
        self.session_colors = ['#00F2FF', '#FF007F', '#70FF00', '#FFD700', '#A020F0']
        self.plots = [] 
        self.all_ghost_curves = [] # 박제 보관함
        self.current_active_curves = [] 
        self.current_color_idx = -1
        self.x_range = np.linspace(0, 1, 150)
        
        # [Peak Hold용 변수]
        self.max_error_seen = threshold * 3

        self._build_4col_layout()
        self._setup_error_trace_layout(threshold)
        self.start_new_session()

    def _build_4col_layout(self):
        """기존 4열(또는 8열) 레이아웃 유지"""
        cols_per_row = 5 # 5열이 시인성이 좋아 수정했습니다. 원하시면 8로 바꾸셔도 됩니다.
        for i, f_name in enumerate(self.features):
            p = self.win.addPlot()
            p.setFixedHeight(180)
            self._apply_plot_style(p, f_name)
            self.plots.append(p)
            if (i + 1) % cols_per_row == 0:
                self.win.nextRow()
        if self.num_features % cols_per_row != 0:
            self.win.nextRow()

    def _apply_plot_style(self, p, title):
        p.showGrid(x=True, y=True, alpha=0.1)
        p.setXRange(0, 1)
        p.setYRange(0, 1.1)
        p.setTitle(f"<span style='color: #4ECDC4; font-size: 9pt; font-family: Consolas;'>{title.upper()}</span>")
        ax = p.getAxis('bottom'); ax.setStyle(showValues=False)
        p.getAxis('left').setStyle(showValues=False)
        p.setMenuEnabled(False)
        p.addItem(pg.InfiniteLine(pos=0.5, angle=90, pen=pg.mkPen('#333333', width=1, style=Qt.PenStyle.DashLine)))

    def _setup_error_trace_layout(self, threshold):
        self.win.addLabel("<br><b><span style='color: #FF4444; font-size: 11pt;'>🚨 ANOMALY SCORE</span></b>", colspan=5)
        self.win.nextRow()
        self.status_plot = self.win.addPlot(colspan=5)
        self.status_plot.setFixedHeight(200)
        
        # [Y축 0 고정 및 Peak 추적 설정]
        vbox = self.status_plot.getViewBox()
        vbox.setLimits(yMin=0) # 0 아래로 안 내려가게 박제
        self.status_plot.setYRange(0, self.max_error_seen, padding=0)
        
        self.error_history = []
        self.thresh_line = pg.InfiniteLine(pos=threshold, angle=0, pen=pg.mkPen('#FF4444', width=2))
        self.status_plot.addItem(self.thresh_line)

    def start_new_session(self):
        """원래 코드의 박제(Alpha 변경) 로직 복구"""
        if self.current_active_curves:
            for curve in self.current_active_curves:
                c = pg.mkColor(curve.opts['pen'].color())
                c.setAlpha(90) # 흐릿하게 박제
                curve.setPen(pg.mkPen(c, width=1.0))
                self.all_ghost_curves.append(curve)

        self.current_color_idx = (self.current_color_idx + 1) % len(self.session_colors)
        color = self.session_colors[self.current_color_idx]
        
        # 새 커브들을 생성 (기존 plots에 겹쳐서 그려짐 = 박제 효과)
        self.current_active_curves = [p.plot(pen=pg.mkPen(color, width=2.5)) for p in self.plots]
        self.error_history = []
        self.current_error_curve = self.status_plot.plot(pen=pg.mkPen(color, width=2.0))

    def update_view(self, current_features, avg_error, current_threshold):
        if current_features is None: return
        if isinstance(current_features, str) and "NEW_SESSION" in current_features:
            self.start_new_session(); return

        try:
            for i, val in enumerate(current_features):
                if i < len(self.plots):
                    self._draw_gaussian_stat(i, val)

            if avg_error is not None:
                err_val = float(avg_error)
                self.error_history.append(err_val)
                if len(self.error_history) > 500: self.error_history.pop(0)
                
                # [Peak Hold 로직]
                if err_val > self.max_error_seen:
                    self.max_error_seen = err_val * 1.2
                    self.status_plot.setYRange(0, self.max_error_seen, padding=0)
                
                self.current_error_curve.setData(self.error_history)
                self.thresh_line.setValue(current_threshold)
                
                bg = (70, 0, 0, 80) if err_val > current_threshold else (10, 10, 12, 255)
                self.status_plot.getViewBox().setBackgroundColor(bg)
        except Exception:
            pass

    def _draw_gaussian_stat(self, plot_idx, val):
        f_name = self.features[plot_idx]
        
        # 기본값 설정
        mu_visual = 0.5   # 차트 중앙 (0점)
        sig_visual = 0.07 # 기본 폭
        
        # 1. 평균(mean) 지표일 경우: 위치(mu)를 적극적으로 이동
        if "mean" in f_name:
            # 보통 물리량 평균은 0보다 큰 경우가 많으므로 범위를 적절히 조절
            # 예: [-10, 10] 범위를 [0.1, 0.9] 시각적 영역으로
            mu_visual = np.interp(val, [-10, 10], [0.1, 0.9])
            
        # 2. 표준편차(std) 지표일 경우: 종의 폭(sigma)을 조절
        elif "std" in f_name:
            # std가 커질수록 종이 옆으로 퍼지게 설정 (0.05 ~ 0.2)
            sig_visual = np.interp(val, [0, 5], [0.05, 0.2])
            # std 차트 자체는 중앙(0.5)에 고정하거나 mean과 연동 가능
            mu_visual = 0.5 
            
        # 3. 그 외 (skew, rough, entropy 등): 기존처럼 위치 이동
        else:
            mu_visual = np.interp(val, [-10, 10], [0.1, 0.9])
            sig_visual = 0.07 # 고정 폭

        # 최종 클리핑 및 가우시안 계산
        mu_visual = np.clip(mu_visual, 0.01, 0.99)
        gauss = np.exp(-0.5 * ((self.x_range - mu_visual) / sig_visual)**2)
        
        # 차트 업데이트
        self.current_active_curves[plot_idx].setData(self.x_range, gauss)