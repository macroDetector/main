import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
import sys

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
        self.win.resize(1200, 950)

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

        