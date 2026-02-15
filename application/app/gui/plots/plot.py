import sys
import pyqtgraph as pg

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer
from multiprocessing import Queue

def plot_main(points, interval=10, log_queue: Queue = None):
    if not points:
        print('DB에 저장된 point가 없습니다.')
        return

    if log_queue:
        log_queue.put("PyQtGraph Plot 실행")

    # 1. QApplication 생성 (PyQt6 방식)
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # 2. 메인 윈도우 및 그래프 설정
    win = QMainWindow()
    win.setWindowTitle("Mouse Movement Path (PyQtGraph)")
    win.resize(800, 600)

    central_widget = QWidget()
    win.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)

    plot_widget = pg.PlotWidget()
    layout.addWidget(plot_widget)

    plot_widget.setBackground('#1F2024')
    plot_widget.showGrid(x=True, y=True, alpha=0.3)
    plot_widget.setLabel('left', 'Y Coordinate')
    plot_widget.setLabel('bottom', 'X Coordinate')
    
    plot_widget.invertY(True)

    line_item = plot_widget.plot(pen=pg.mkPen(color='#FF6F61', width=2))
    scatter_item = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(255, 255, 255, 150))
    plot_widget.addItem(scatter_item)

    state = {'idx': 0, 'move_x': [], 'move_y': []}

    def update():
        idx = state['idx']
        if idx < len(points):
            p = points[idx]
            
            if isinstance(p, dict):
                x, y = p.get("x", 0), p.get("y", 0)
            else:
                x, y = getattr(p, "x", 0), getattr(p, "y", 0)

            state['move_x'].append(x)
            state['move_y'].append(y)

            line_item.setData(state['move_x'], state['move_y'])
            scatter_item.setData(pos=list(zip(state['move_x'], state['move_y'])))

            if idx == 0:
                 plot_widget.setXRange(x - 100, x + 100)
                 plot_widget.setYRange(y - 100, y + 100)
            else:
                 plot_widget.enableAutoRange(enable=True)

            state['idx'] += 1
        else:
            timer.stop()
            if log_queue:
                log_queue.put("[Process] 모든 좌표 출력 완료")

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(interval)

    win.show()
    
    def on_exit():
        if log_queue:
            log_queue.put("[Process] Plot 창 종료")
    
    # PyQt6에서는 aboutToQuit 연결 방식이 동일하지만, 
    # 가급적 한 엔진만 쓰도록 유도
    app.aboutToQuit.connect(on_exit)
    
    # PyQt6에서는 exec_() 대신 exec() 사용 가능 (3.10+ 기준)
    app.exec()