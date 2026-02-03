import torch
import joblib
import numpy as np
import pandas as pd
from collections import deque
from multiprocessing import Queue, Event

import app.core.globals as g_vars
from app.models.TransformerMacroDetector import TransformerMacroAutoencoder
from app.services.indicators import indicators_generation

def inferece_plot_main(chart_queue: Queue, features, threshold, stop_event=None):
    import sys
    from app.services.RealTimeMonitor import RealTimeMonitor
    from PyQt6.QtCore import QTimer
    
    if stop_event is None:
        stop_event = Event()

    monitor = RealTimeMonitor(features, threshold)
    
    def update():
        if stop_event.is_set():
            timer.stop()
            monitor.app.quit()
            return

        try:
            while not chart_queue.empty():
                data = chart_queue.get_nowait()
                # data: (tensor_np, error, current_threshold)
                if len(data) == 3:
                    monitor.update_view(data[0], data[1], data[2])
                else:
                    monitor.update_view(data[0], data[1], threshold)
        except (EOFError, BrokenPipeError, ConnectionResetError):
            timer.stop()
            monitor.app.quit()
        except Exception:
            pass
                
    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(16)
    sys.exit(monitor.app.exec())

class MacroDetector:
    def __init__(self, model_path: str, seq_len=g_vars.SEQ_LEN, threshold=None, device=None, chart_Show=True, stop_event=None):
        self.seq_len = seq_len
        # globals에 정의된 최적의 threshold를 그대로 사용합니다.
        self.base_threshold = threshold if threshold is not None else g_vars.threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 확률 계산용 버퍼 (최근 흐름 반영)
        self.smooth_error_buf = deque(maxlen=7) 
        self.stop_event = stop_event or Event()
        self.chart_Show = chart_Show
        self.plot_proc = None

        # ===== 모델 초기화 =====
        self.model = TransformerMacroAutoencoder(
            input_size=len(g_vars.FEATURES),
            d_model=g_vars.d_model,
            nhead=4,
            num_layers=g_vars.num_layers,
            dim_feedforward=128,
            dropout=g_vars.dropout
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.scaler = joblib.load(g_vars.scaler_path)
        self.buffer = deque(maxlen=seq_len * 3)

    def start_plot_process(self):
        if not self.chart_Show or (self.plot_proc and self.plot_proc.is_alive()):
            return

        from multiprocessing import Process
        self.plot_proc = Process(
            target=inferece_plot_main,
            args=(g_vars.CHART_DATA, g_vars.FEATURES, self.base_threshold, self.stop_event),
            daemon=False
        )
        self.plot_proc.start()

    def push(self, data: dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))
        if len(self.buffer) < self.seq_len:
            return None
        return self._infer()

    def _infer(self):
        # 1. 데이터 준비
        df = pd.DataFrame(list(self.buffer), columns=["x", "y", "timestamp", "deltatime"])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = indicators_generation(df)

        # 2. Feature 추출 및 클리핑
        df_features = df[g_vars.FEATURES].tail(self.seq_len).copy()
        if g_vars.CLIP_BOUNDS:
            for col, b in g_vars.CLIP_BOUNDS.items():
                if col in df_features.columns:
                    df_features[col] = df_features[col].clip(lower=b['min'], upper=b['max'])

        # 3. 스케일링 및 추론
        try:
            X_scaled = self.scaler.transform(df_features.values)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(X_tensor)
                recon_error = torch.abs(output - X_tensor).mean().item()
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            return None

        # 4. 확률 시스템 핵심 (Sigmoid Probability)
        self.smooth_error_buf.append(recon_error)
        avg_error = np.mean(self.smooth_error_buf)

        # 에러가 base_threshold일 때 정확히 50%가 나오도록 설계
        # sensitivity(20.0)가 높을수록 threshold 근처에서 확률이 급격하게 변함
        sensitivity = 20.0 
        diff = avg_error - self.base_threshold
        macro_prob = 1 / (1 + np.exp(-sensitivity * diff))
        macro_score = round(macro_prob * 100, 2)

        # 5. 최종 판정 (확률 80% 이상이면 매크로 의심)
        is_human = macro_score < 80.0

        # 6. 모니터링 데이터 전송
        if g_vars.CHART_DATA is not None:
            try:
                g_vars.CHART_DATA.put_nowait((X_tensor.cpu().numpy(), avg_error, self.base_threshold))
            except: pass

        return {
            "is_human": is_human,
            "macro_probability": f"{macro_score}%",
            "prob_value": macro_score,
            "raw_error": round(avg_error, 5),
            "threshold": self.base_threshold
        }