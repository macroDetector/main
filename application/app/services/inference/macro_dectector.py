import torch
import joblib
import numpy as np
import pandas as pd
from collections import deque
from multiprocessing import Queue, Event
import sys

from sklearn.preprocessing import RobustScaler
import app.core.globals as g_vars
from app.models.TransformerMacroDetector import TransformerMacroAutoencoder
from app.core.indicators import indicators_generation

from app.utilites.make_sequence import make_seq
from app.utilites.make_gauss import make_gauss
from app.utilites.loss_caculation import Loss_Calculation

def inferece_plot_main(chart_queue: Queue, features, threshold, chart_view, process_lock, stop_event=None):
    from app.utilites.plot_monitor import RealTimeMonitor
    from PyQt6.QtCore import QTimer
    

    exit_code = 1
    if stop_event is None:
        stop_event = Event()

    try:
        monitor = RealTimeMonitor(features, threshold)
        
        def update():
            if stop_event.is_set():
                with process_lock:
                    chart_view.value = False            
                monitor.app.quit()
                return

            try:
                last_data = None
                while not chart_queue.empty():
                    last_data = chart_queue.get_nowait()

                if last_data is not None:
                    if isinstance(last_data, str):
                        if last_data == "NEW_SESSION":
                            monitor.update_view("NEW_SESSION", None, None)
                    else:
                        if len(last_data) == 3:
                            monitor.update_view(last_data[0], last_data[1], last_data[2])
                        else:
                            monitor.update_view(last_data[0], last_data[1], threshold)
                            
            except (EOFError, BrokenPipeError, ConnectionResetError):
                monitor.app.quit()
            except Exception as e:
                # print(f"Update Loop Error: {e}")
                pass
                    
        timer = QTimer()
        timer.timeout.connect(update)
        timer.start(16)

        exit_code = monitor.app.exec()

    except Exception as e:
        print(f"❌ 차트 프로세스 에러: {e}")
    finally:
        with process_lock:
            chart_view.value = False
        print(f"✅ 차트 리소스 반납 완료 {chart_view.value}")

        sys.exit(exit_code)
    

class MacroDetector:
    def __init__(
            self, 
            model_path: str, 
            scale_path:str, 
            seq_len=g_vars.SEQ_LEN, 
            threshold=None, 
            device=None, 
            chart_Show=True, 
            stop_event=None,
            log_queue:Queue=None):
        
        self.seq_len = seq_len
        self.base_threshold = threshold if threshold is not None else g_vars.threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_queue = log_queue
        # 안정 장치
        self.buffer = deque(maxlen=10)

        self.stop_event = stop_event
        self.chart_Show = chart_Show
        self.plot_proc = None

        # ===== 모델 초기화 =====
        self.model = TransformerMacroAutoencoder(
            input_size=g_vars.input_size,
            d_model=g_vars.d_model,
            nhead=g_vars.n_head,
            num_layers=g_vars.num_layers,
            dim_feedforward=g_vars.dim_feedforward,
            dropout=g_vars.dropout
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.scaler:RobustScaler = joblib.load(scale_path)

    def push(self, data: dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))
        
    def start_plot_process(self):
        """실시간 차트 프로세스를 시작합니다."""
        if not self.chart_Show or (self.plot_proc and self.plot_proc.is_alive()):
            return

        from multiprocessing import Process

        self.plot_proc = Process(
            target=inferece_plot_main, 
            kwargs={
                "chart_queue" : g_vars.CHART_DATA, 
                "features" : g_vars.FEATURES, 
                "threshold" : self.base_threshold, 
                "stop_event" : self.stop_event, 
                "chart_view" : g_vars.INFERENCE_CHART_VIEW,
                "process_lock" : g_vars.PROCESS_LOCK
            },
            daemon=False
        )
        self.plot_proc.start()

    def _infer(self):
        df = pd.DataFrame(list(self.buffer), columns=["x", "y", "timestamp", "deltatime"])
    
        df = df[df["deltatime"] <= g_vars.filter_tolerance].reset_index(drop=True)
        
        df = indicators_generation(
            df_chunk=df, 
            chunk_size=g_vars.chunk_size,
            offset=g_vars.offset
        )
        
        if len(df) < g_vars.SEQ_LEN:
            return None
                
        df_filter_chunk = df[g_vars.FEATURES].copy()
        
        chunks_scaled_array = self.scaler.transform(df_filter_chunk)
        
        chunks_scaled_df = pd.DataFrame(chunks_scaled_array, columns=g_vars.FEATURES)
        
        chunks_scaled_df = chunks_scaled_df * g_vars.scale_array
  
        final_input:np.array = make_seq(data=chunks_scaled_df, seq_len=g_vars.SEQ_LEN, stride=1)
        
        send_data = []
        for i, input in enumerate(final_input):
            if self.stop_event.is_set():
                if self.log_queue:
                    self.log_queue.put("🛑 Detector 중지")
                else:
                    print("🛑 Detector 중지")
                break            
            last_seq = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(last_seq)

                sample_errors = Loss_Calculation(outputs=output, batch=last_seq).mean().item()

                # 임계치 판정 logic
                is_human = sample_errors <= self.base_threshold
                
            if g_vars.CHART_DATA is not None:
                try:
                    # chunks_scaled_df에서 현재 시퀀스의 '끝 지점' 데이터를 추출하여 전송
                    # 보통 시퀀스의 마지막 타임스텝 데이터를 특징값으로 봅니다.
                    current_features = chunks_scaled_df.iloc[i + g_vars.SEQ_LEN - 1] 
                    g_vars.CHART_DATA.put_nowait((current_features, sample_errors, self.base_threshold))
                except Exception: 
                    pass

            _error = sample_errors / self.base_threshold * 100
                    
            send_data.append({
                "is_human": is_human,
                "error_pct": _error, 
            })
            
        
            # 메시지 구성
            log_text = f"{is_human}, {_error:.4f} %"

            self.log_queue.put(log_text) if self.log_queue else print(log_text)

        return send_data