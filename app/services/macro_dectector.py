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
        self.base_threshold = threshold if threshold is not None else g_vars.threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # [변경] 최근 100개 데이터 포인트만 유지 (연산 효율화)
        self.buffer = deque(maxlen=100) 
        
        # 노이즈 방지를 위해 최근 3~5개 에러의 평균만 사용 (순간적인 튐 방지)
        self.smooth_error_buf = deque(maxlen=5) 
        
        self.stop_event = stop_event or Event()
        self.chart_Show = chart_Show
        self.plot_proc = None

        # ===== 모델 초기화 =====
        self.model = TransformerMacroAutoencoder(
            input_size=len(g_vars.FEATURES),
            d_model=g_vars.d_model,
            nhead=g_vars.n_head,
            num_layers=g_vars.num_layers,
            dim_feedforward=g_vars.dim_feedforward,
            dropout=g_vars.dropout
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.scaler = joblib.load(g_vars.scaler_path)

    def push(self, data: dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))
        
        # 최소 seq_len은 채워져야 분석 시작
        if len(self.buffer) < self.seq_len:
            return None
        return self._infer()

    def start_plot_process(self):
        """실시간 차트 프로세스를 시작합니다."""
        if not self.chart_Show or (self.plot_proc and self.plot_proc.is_alive()):
            return

        from multiprocessing import Process
        # inferece_plot_main는 파일 상단에 정의되어 있어야 합니다.
        self.plot_proc = Process(
            target=inferece_plot_main, 
            args=(g_vars.CHART_DATA, g_vars.FEATURES, self.base_threshold, self.stop_event),
            daemon=False
        )
        self.plot_proc.start()

    def _infer(self):
        # 1. 최근 100개 데이터로 피처 생성
        df = pd.DataFrame(list(self.buffer), columns=["x", "y", "timestamp", "deltatime"])
        df = indicators_generation(df)

        # 2. 모델 입력용 마지막 seq_len 추출
        df_features = df[g_vars.FEATURES].tail(self.seq_len).copy()
        
        if g_vars.CLIP_BOUNDS:
            for col, b in g_vars.CLIP_BOUNDS.items():
                if col in df_features.columns:
                    df_features[col] = df_features[col].clip(lower=b['min'], upper=b['max'])

        try:
            X_scaled = self.scaler.transform(df_features.values)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(X_tensor)
                # 재구성 에러 (Reconstruction Error)
                recon_error = torch.mean((output - X_tensor)**2).item()
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            return None

        # 3. 에러 스무딩 (너무 민감하게 반응하지 않도록 최근 5개 평균)
        self.smooth_error_buf.append(recon_error)
        avg_error = np.mean(self.smooth_error_buf)

        # 4. [변경] 단순 Threshold 판정 로직
        # 평균 에러가 설정한 임계값을 넘으면 바로 매크로(False) 판정
        is_human = avg_error < self.base_threshold * 1.05
        
        # 시각적인 확률 표기 (단순히 에러/임계값 비율로 표시)
        macro_score = min(100.0, round((avg_error / self.base_threshold) * 50, 2))
        if not is_human:
            # 임계값을 넘는 순간 50~100 사이로 표기
            macro_score = min(100.0, 50.0 + (avg_error - self.base_threshold) * 100)

        # 5. 모니터링 데이터 전송
        if g_vars.CHART_DATA is not None:
            try:
                g_vars.CHART_DATA.put_nowait((X_tensor.cpu().numpy(), avg_error, self.base_threshold))
            except: pass

        return {
            "is_human": is_human,
            "macro_probability": f"{'🚨 MACRO' if not is_human else '🙂 HUMAN'}",
            "prob_value": macro_score,
            "raw_error": round(avg_error, 5),
            "threshold": self.base_threshold
        }