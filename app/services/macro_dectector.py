import torch
import joblib

from collections import deque

from app.models.TransformerMacroDetector import TransformerMacroAutoencoder

from app.services.indicators import indicators_generation

import pandas as pd
import app.core.globals as globals
from multiprocessing import Queue

from multiprocessing import Process
    
from app.utilites.points_to_features import points_to_features
from multiprocessing import Event

def inferece_plot_main(chart_queue: Queue, features, threshold, stop_event=None):
    import sys
    from app.services.RealTimeMonitor import RealTimeMonitor
    from PyQt6.QtCore import QTimer
    
    if stop_event is None:
        from multiprocessing import Event
        stop_event = Event()

    monitor = RealTimeMonitor(features, threshold)
    
    def update():
        # 1. 큐 확인보다 '중지 이벤트'를 최우선으로 체크
        if stop_event.is_set():
            timer.stop()
            monitor.app.quit()
            return

        # 2. 파이프가 깨졌을 때(BrokenPipe)를 대비한 try-except 강화
        try:
            # empty() 호출 시에도 파이프 에러가 날 수 있으므로 안으로 밀어넣음
            while not chart_queue.empty():
                data = chart_queue.get_nowait()
                monitor.update_view(data[0], data[1])
        except (EOFError, BrokenPipeError, ConnectionResetError):
            # 메인 프로세스가 죽어서 파이프가 끊긴 경우
            timer.stop()
            monitor.app.quit()
        except Exception:
            # 기타 큐가 비어있는 등의 예외는 무시하고 계속 진행
            pass
                
    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(16) 
    
    sys.exit(monitor.app.exec())
class MacroDetector:
    def __init__(self, model_path: str, seq_len=globals.SEQ_LEN, threshold=0.8, device=None, chart_Show=True, stop_event=None):
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.stop_event = stop_event
        if not self.stop_event:
            self.stop_event = Event()

        self.chart_Show = chart_Show
        self.plot_proc = None

        # ===== 모델 초기화 =====
        self.model = TransformerMacroAutoencoder(
            input_size=len(globals.FEATURES),
            d_model=globals.d_model,
            nhead=4,
            num_layers=globals.num_layers,
            dim_feedforward=128,
            dropout=globals.dropout
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        self.scaler = joblib.load(globals.scaler_path)

        # ===== 좌표 buffer =====
        self.buffer = deque(maxlen=seq_len * 3)
        self.prev_speed = 0.0

    def start_plot_process(self):
        if not self.chart_Show:
            return

        if self.plot_proc and self.plot_proc.is_alive():
            return

        from multiprocessing import Process
        self.plot_proc = Process(
            target=inferece_plot_main,
            args=(
                globals.CHART_DATA,
                globals.FEATURES,
                self.threshold,
                self.stop_event
            ),
            daemon=False
        )
        self.plot_proc.start()

    def push(self, data:dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))

        if len(self.buffer) < self.seq_len * 3:
            return None
        
        return self._infer()

    def _infer(self):
        xs = [p[0] for p in self.buffer]
        ys = [p[1] for p in self.buffer]
        ts = [p[2] for p in self.buffer] 
        deltatime = [p[3] for p in self.buffer] 

        df = pd.DataFrame({"timestamp": ts, "x": xs, "y": ys, "deltatime" : deltatime})
        df = indicators_generation(df)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[globals.FEATURES].copy()
        
        if len(df) < self.seq_len:
            return None
        
        # ===== Feature 추출 =====
        X_infer, _ = points_to_features(df_chunk=df, seq_len=self.seq_len, stride=globals.STRIDE)

        # [수정 2] points_to_features 결과가 비어있거나 차원이 맞지 않는 경우 체크
        if X_infer is None or X_infer.size == 0 or len(X_infer.shape) < 3:
            return None

        # 이제 안전하게 shape를 가져올 수 있습니다.
        n_infer, seq_len, n_features = X_infer.shape

        # 스케일링 적용
        X_infer_reshaped = X_infer.reshape(-1, n_features)
        X_infer_scaled = self.scaler.transform(X_infer_reshaped)
        X_infer = X_infer_scaled.reshape(n_infer, seq_len, n_features)
        
        # 마지막 시퀀스만 사용
        X_tensor = torch.tensor(X_infer[-1], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # labeling
        with torch.no_grad():
            X_recon = self.model(X_tensor)
            recon_error = torch.abs(X_recon - X_tensor).mean(dim=(1,2)).cpu().item()
        
        if globals.CHART_DATA is not None:
            globals.CHART_DATA.put((X_tensor.cpu().numpy(), recon_error))

        return {"is_human": recon_error < self.threshold, "prob": recon_error}