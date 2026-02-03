import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt  # DashLine 설정을 위해 필요
import numpy as np
import sys

class RealTimeMonitor:
    def __init__(self, features, threshold, window_size=100):
        self.app = QApplication.instance() or QApplication(sys.argv)
            
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Feature Monitor")
        self.win.resize(1200, 800) # 가로를 좀 더 넓게 조절
        
        self.features = features
        self.window_size = window_size
        self.plots = []
        self.curves = []
        
        self.data = np.zeros((len(features) + 2, window_size))
        
        # --- Feature별 그래프 생성 (3열 배치) ---
        num_cols = 3
        for i, f in enumerate(features):
            row_idx = i // num_cols
            col_idx = i % num_cols
            
            p = self.win.addPlot(row=row_idx, col=col_idx, title=f)
            p.showGrid(x=True, y=True)
            # 텍스트 크기 조절 (칸이 좁아지므로)
            p.setTitle(f, size='10pt') 
            
            curr_curve = p.plot(pen='c') 
            self.plots.append(p)
            self.curves.append(curr_curve)
            
        # --- Reconstruction Error 그래프 (가장 아래에 꽉 차게) ---
        # 마지막 피처가 있던 줄 다음 줄(last_row) 계산
        last_row = (len(features) - 1) // num_cols + 1
        
        # col=0부터 시작해서 3칸을 다 차지하도록(colspan=3) 설정
        self.error_plot = self.win.addPlot(row=last_row, col=0, colspan=num_cols, 
                                           title="Reconstruction Error (Dynamic Threshold)")
        self.error_plot.showGrid(x=True, y=True)
        self.error_plot.setFixedHeight(250) # 에러 차트는 중요하니 높이 고정
        
        self.error_curve = self.error_plot.plot(pen=pg.mkPen('r', width=2))
        self.threshold_curve = self.error_plot.plot(
            pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine)
        )
        
    def update_view(self, x_tensor_np, error, current_threshold):
        """
        x_tensor_np: 모델 입력 데이터
        error: 계산된 리컨스트럭션 에러
        current_threshold: 계산된 동적 임계치
        """
        # 데이터 롤링 (왼쪽으로 한 칸씩 밀기)
        self.data[:, :-1] = self.data[:, 1:]
        
        # 새로운 데이터 삽입
        current_vals = x_tensor_np[0, -1, :] # 최신 시퀀스의 마지막 값
        self.data[:len(self.features), -1] = current_vals
        self.data[-2, -1] = error            # 에러값 저장
        self.data[-1, -1] = current_threshold # 임계치값 저장
        
        # 각 피처 그래프 업데이트
        for i in range(len(self.features)):
            self.curves[i].setData(self.data[i])
            
        # 에러 및 임계치 그래프 업데이트
        self.error_curve.setData(self.data[-2])
        self.threshold_curve.setData(self.data[-1])

class TrainMonitor:
    def __init__(self, window_size=500):
        # QApplication 보장
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)

        self.win = pg.GraphicsLayoutWidget(
            show=True,
            title="Model Training Monitor"
        )
        self.win.resize(900, 450)

        self.window_size = window_size

        # 데이터 (list 사용 → 스케일 버그 원천 차단)
        self.epochs = []
        self.train_loss_data = []
        self.val_loss_data = []

        # Plot
        self.p1 = self.win.addPlot(title="Real-time Training & Validation Loss")
        self.p1.setLabel('left', 'Loss (MSE)')
        self.p1.setLabel('bottom', 'Epoch')
        self.p1.addLegend()
        self.p1.showGrid(x=True, y=True)

        self.p1.getAxis('left').enableAutoSIPrefix(False)
        
        # Curves
        self.train_curve = self.p1.plot(
            pen=pg.mkPen('#00E5FF', width=2),
            name="Train Loss"
        )
        self.val_curve = self.p1.plot(
            pen=pg.mkPen('#FF4B4B', width=2),
            name="Val Loss"
        )

        # 초기 autoscale 활성
        self.p1.enableAutoRange(axis='y', enable=True)

    def update_view(self, epoch, train_loss, val_loss):
        # 안전하게 타입 보장
        epoch = int(epoch)
        train_loss = float(train_loss)
        val_loss = float(val_loss)

        self.epochs.append(epoch)
        self.train_loss_data.append(train_loss)
        self.val_loss_data.append(val_loss)

        # Rolling window 유지
        if len(self.epochs) > self.window_size:
            self.epochs = self.epochs[-self.window_size:]
            self.train_loss_data = self.train_loss_data[-self.window_size:]
            self.val_loss_data = self.val_loss_data[-self.window_size:]

        # Plot 업데이트
        self.train_curve.setData(self.epochs, self.train_loss_data)
        self.val_curve.setData(self.epochs, self.val_loss_data)

        # y축 자동 재조정 (착시 방지 핵심)
        self.p1.enableAutoRange(axis='y', enable=True)