import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt  # DashLine 설정을 위해 필요
import numpy as np
import sys

class RealTimeMonitor:
    def __init__(self, features, threshold, window_size=100):
        # PySide6 -> PyQt6
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
            
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Feature Monitor")
        self.win.resize(1000, 800)
        
        self.features = features
        self.window_size = window_size
        self.plots = []
        self.curves = []
        
        self.data = np.zeros((len(features) + 1, window_size))
        
        for i, f in enumerate(features):
            p = self.win.addPlot(row=i, col=0, title=f)
            p.showGrid(x=True, y=True)
            curr_curve = p.plot(pen='c') 
            self.plots.append(p)
            self.curves.append(curr_curve)
            
        self.error_plot = self.win.addPlot(row=len(features), col=0, title="Reconstruction Error")
        
        # pg.QtCore.Qt.DashLine -> Qt.PenStyle.DashLine (PyQt6 방식)
        self.error_plot.addLine(y=threshold, pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine))
        self.error_curve = self.error_plot.plot(pen=pg.mkPen('r', width=2))

    def update_view(self, x_tensor_np, error):
        current_vals = x_tensor_np[0, -1, :]
        self.data[:, :-1] = self.data[:, 1:]
        self.data[:-1, -1] = current_vals
        self.data[-1, -1] = error
        
        for i in range(len(self.features)):
            self.curves[i].setData(self.data[i])
        self.error_curve.setData(self.data[-1])

class TrainMonitor:
    def __init__(self, window_size=500):
        # PyQt6 App 초기화
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
            
        self.win = pg.GraphicsLayoutWidget(show=True, title="Model Training Monitor")
        self.win.resize(800, 400)
        
        self.window_size = window_size
        self.train_loss_data = np.zeros(window_size)
        self.val_loss_data = np.zeros(window_size)
        self.ptr = 0

        # Loss Plot 설정
        self.p1 = self.win.addPlot(title="Real-time Training & Validation Loss")
        self.p1.setLabel('left', 'Loss', units='MSE')
        self.p1.setLabel('bottom', 'Epochs')
        self.p1.addLegend()
        self.p1.showGrid(x=True, y=True)

        # 커브 설정 (Train: Cyan, Val: Red)
        self.train_curve = self.p1.plot(pen=pg.mkPen('#00E5FF', width=2), name="Train Loss")
        self.val_curve = self.p1.plot(pen=pg.mkPen('#FF4B4B', width=2), name="Val Loss")

    def update_view(self, train_loss, val_loss):
        """Queue에서 받은 데이터를 그래프에 업데이트"""
        if self.ptr < self.window_size:
            self.train_loss_data[self.ptr] = train_loss
            self.val_loss_data[self.ptr] = val_loss
            self.ptr += 1
        else:
            # 창이 꽉 차면 밀어내기 (Rolling Window)
            self.train_loss_data[:-1] = self.train_loss_data[1:]
            self.train_loss_data[-1] = train_loss
            self.val_loss_data[:-1] = self.val_loss_data[1:]
            self.val_loss_data[-1] = val_loss

        self.train_curve.setData(self.train_loss_data[:self.ptr])
        self.val_curve.setData(self.val_loss_data[:self.ptr])