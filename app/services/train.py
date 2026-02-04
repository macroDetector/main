import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time
import numpy as np
import json


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import app.core.globals as g_vars
import joblib
# 모델
from app.models.MousePoint import MousePoint
from app.models.TransformerMacroDetector import TransformerMacroAutoencoder, MacroDataset

import app.repostitories.DBController as DBController
import app.repostitories.JsonController as JsonController

from app.services.indicators import indicators_generation
from multiprocessing import Queue

from app.utilites.make_df_from_points import make_df_from_points
from app.utilites.points_to_features import points_to_features
from app.utilites.save_confing import update_parameters

def train_plot_main(train_queue: Queue):
    import sys
    from app.services.RealTimeMonitor import TrainMonitor
    from PyQt6.QtCore import QTimer

    monitor = TrainMonitor(window_size=1000)

    def update():
        while not train_queue.empty():
            epoch, train_loss, val_loss = train_queue.get_nowait()
            monitor.update_view(epoch, train_loss, val_loss)

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(100)

    sys.exit(monitor.app.exec())


class TrainMode():
    def __init__(self, stop_event=None, log_queue:Queue=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.stop_event=stop_event
        self.seq_len=g_vars.SEQ_LEN
        self.log_queue:Queue=log_queue

        self.plot_proc = None


    def start_plot_process(self):
        if self.plot_proc is not None and self.plot_proc.is_alive():
            return

        from multiprocessing import Process
        self.plot_proc = Process(
            target=train_plot_main,
            args=(g_vars.TRAIN_DATA,),
            daemon=False
        )
        self.plot_proc.start()

    # train
    def train_start(self, train_dataset, val_dataset, batch_size=g_vars.batch_size, epochs=2000, lr=g_vars.lr,
                    device=None, model=None, stop_event=None, patience=20, log_queue=None, save_path="best_model.pth"):
        try:
            log_queue.put(f"lr : {lr}, batch_size : {batch_size}")
            self.start_plot_process()

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr,
                weight_decay=g_vars.weight_decay
            )

            best_val_loss = float('inf')
            epochs_no_improve = 0
            # 5% 개선을 기준으로 설정 (필요에 따라 0.05 ~ 0.20 조절)
            min_improvement_factor = 0.05 
            recommended_base_threshold = 0.0

            for epoch in range(epochs):
                if stop_event.is_set():
                    self.train_stop_event(log_queue=log_queue)
                    break

                # ===== Train Phase =====
                model.train()
                total_train_loss = 0
                for batch_x in train_loader:
                    batch_x = batch_x.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_train_loss += loss.item() * batch_x.size(0)

                avg_train_loss = total_train_loss / len(train_dataset)

                # ===== Validation Phase =====
                model.eval()
                total_val_loss = 0
                all_val_errors = [] 

                with torch.no_grad():
                    for batch_x in val_loader:
                        batch_x = batch_x.to(device)
                        outputs = model(batch_x)
                        
                        loss = criterion(outputs, batch_x)
                        total_val_loss += loss.item() * batch_x.size(0)
                        
                        # MSE 기반 임계치 계산용 데이터 수집
                        sample_errors = torch.mean((outputs - batch_x)**2, dim=(1, 2)) 
                        all_val_errors.extend(sample_errors.cpu().numpy())

                avg_val_loss = total_val_loss / len(val_dataset)
                current_epoch_threshold = np.percentile(all_val_errors, 95)

                # ===== Logging & Early Stopping Logic =====
                status_msg = f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Est.Thresh: {current_epoch_threshold:.6f}"
                if log_queue: log_queue.put(status_msg)

                # 시각화 데이터 큐 전달
                if g_vars.TRAIN_DATA is not None:
                   g_vars.TRAIN_DATA.put((epoch + 1, float(avg_train_loss), float(avg_val_loss)))

                # 개선율 계산
                improvement = 0
                if best_val_loss != float('inf'):
                    improvement = (best_val_loss - avg_val_loss) / best_val_loss

                # [핵심 로직] 첫 에포크이거나, 개선율이 기준치 이상일 때
                if best_val_loss == float('inf') or improvement >= min_improvement_factor:
                    is_initial = (best_val_loss == float('inf'))
                    best_val_loss = avg_val_loss
                    recommended_base_threshold = current_epoch_threshold
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), save_path)
                    
                    if log_queue:
                        msg = "First Best" if is_initial else f"Improvement {improvement*100:.1f}%"
                        log_queue.put(f" >> [Model Saved] {msg} | Loss: {best_val_loss:.6f}")
                else:
                    epochs_no_improve += 1
                    if log_queue:
                        log_queue.put(f" >> [No Significant Improve] Count: {epochs_no_improve}/{patience} (Improv: {improvement*100:.1f}%)")
                    
                    if epochs_no_improve >= patience:
                        if log_queue: log_queue.put(f"Early Stopping: {min_improvement_factor*100}% 이상의 개선이 {patience} epoch 동안 없음.")
                        if stop_event: stop_event.set()
                        break

            # 최종 파라미터 업데이트
            final_msg = f"학습 종료. 최적 Val Loss: {best_val_loss:.6f} | 추천 Base Threshold: {recommended_base_threshold:.6f}"
            if log_queue: log_queue.put(final_msg)
            
            g_vars.threshold = recommended_base_threshold 
            update_parameters({"THRES" : recommended_base_threshold})
            
            with g_vars.lock:
                g_vars.GLOBAL_CHANGE = True

        finally:
            self.train_stop_event(log_queue=log_queue)
            if 'model' in locals() and model is not None:
                model.to('cpu')
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
    def main(self):
        self.log_queue.put(f"device : {self.device} | SEQ_LEN : {g_vars.SEQ_LEN} | STRIDE : {g_vars.STRIDE}")

        # ===== 데이터 읽기 =====
        if g_vars.Recorder == "postgres":
            user_all: list[MousePoint] = DBController.read(True, log_queue=self.log_queue)
            is_dict = False
        elif g_vars.Recorder == "json":
            user_all: list[dict] = JsonController.read(True, log_queue=self.log_queue)
            is_dict = True

        self.log_queue.put(f"user_all length : {len(user_all)}")

        user_df_chunk = make_df_from_points(user_all, is_dict=is_dict)

        user_df_chunk= user_df_chunk.sort_values('timestamp').reset_index(drop=True)

        # ===== Feature 계산 =====
        setting_user_df_chunk: pd.DataFrame = indicators_generation(user_df_chunk)
        
        # ===== Feature 필터 =====
        setting_user_df_chunk = setting_user_df_chunk[g_vars.FEATURES].copy()

        clip_bounds_dict = {}
        for col in g_vars.FEATURES:
            # 1. 일단 콴타일로 범위를 계산합니다.
            lower = float(setting_user_df_chunk[col].quantile(0.05))
            upper = float(setting_user_df_chunk[col].quantile(0.95))
            
            # 2. [핵심 수정] 속도와 관련된 지표들은 '움직임 없음(0)'을 허용해야 합니다.
            # 학습 데이터에 움직임만 있더라도, 추론 시 정지 상태를 위해 min을 0으로 강제합니다.
            if col in ['speed', 'speed_var', 'jerk_std', 'straightness']:
                # straightness는 보통 1이 최소지만 안전하게 0이나 데이터 최소값 중 작은 쪽 선택
                lower = 0.0 
            
            # 3. 각 지표별 특성에 따른 하한선 보정 (필요시)
            # turn, acc 등은 음수가 가능하므로 콴타일 그대로 사용
            
            clip_bounds_dict[col] = {"min": lower, "max": upper}
            setting_user_df_chunk[col] = setting_user_df_chunk[col].clip(lower, upper)

            
        
        # 전역 변수 업데이트
        g_vars.CLIP_BOUNDS = clip_bounds_dict
        update_parameters({"CLIP_BOUNDS" : clip_bounds_dict})
        
        print(f"{json.dumps(clip_bounds_dict, indent=2)} Save")

        print(f"setting_user_df_chunk : {setting_user_df_chunk}")


        if len(setting_user_df_chunk) < g_vars.SEQ_LEN:
            self.log_queue.put("데이터가 충분하지 않습니다.")
            return

        # ===== Train/Val split =====
        user_train_df: pd.DataFrame
        user_val_df: pd.DataFrame

        user_train_df, user_val_df = train_test_split(setting_user_df_chunk, test_size=0.2, shuffle=False)

        # ===== Sequence =====q
        user_train_seq, user_train_pass_seq = points_to_features(df_chunk=user_train_df, seq_len=g_vars.SEQ_LEN, stride=g_vars.STRIDE, log_queue=self.log_queue)

        user_val_seq, user_val_pass_seq = points_to_features(df_chunk=user_val_df, seq_len=g_vars.SEQ_LEN, stride=g_vars.STRIDE, log_queue=self.log_queue)

        self.log_queue.put(f"Length => user_train_seq : {user_train_pass_seq}")
        self.log_queue.put(f"Length => user_val_pass_seq : {user_val_pass_seq}")

        X_train = user_train_seq
        X_val = user_val_seq

        # 스케일링 적용
        n_train, seq_len, n_features = X_train.shape
        n_val = X_val.shape[0]

        scaler = RobustScaler()
        
        # [Train 스케일링] 2D로 펼쳐서 학습(fit) 후 변환(transform)
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_train = X_train_scaled.reshape(n_train, seq_len, n_features)

        # [Val 스케일링] Train 기준 스케일러로 변환만 수행 (중요: fit 안함)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val = X_val_scaled.reshape(n_val, seq_len, n_features)
        joblib.dump(scaler, g_vars.scaler_path)
        
        train_dataset = MacroDataset(X_train)
        val_dataset   = MacroDataset(X_val)
        
        print(f"Total samples: {len(train_dataset)}")

        # 2. 첫 번째 샘플 가져오기
        sample_x = train_dataset[0]

        print("--- First Sample Data ---")
        print(f"Input Shape (seq_len, features): {sample_x.shape}")
        print("--- Raw Input (First 5 steps) ---")
        print(sample_x[:5])

        model = TransformerMacroAutoencoder(
            input_size=len(g_vars.FEATURES),
            d_model=g_vars.d_model,
            nhead=g_vars.n_head,
            num_layers=g_vars.num_layers,
            dim_feedforward=g_vars.dim_feedforward,
            dropout=g_vars.dropout
        ).to(self.device)

        timeinterval = 7

        while timeinterval != 0:
            if self.stop_event.is_set():
                self.train_stop_event(log_queue=self.log_queue)
                return

            timeinterval -= 1
            self.log_queue.put(f"train 시작까지 count : {timeinterval}")

            time.sleep(1)

        self.train_start(
            train_dataset=train_dataset,
            val_dataset=val_dataset, 
            batch_size=g_vars.batch_size, 
            lr=g_vars.lr,
            epochs=g_vars.epoch,
            device=self.device, 
            model=model, 
            stop_event=self.stop_event, 
            patience=g_vars.patience, 
            log_queue=self.log_queue, 
            save_path=g_vars.save_path
        )

    def train_stop_event(self, log_queue: Queue = None):
        # 1. 시각화 프로세스는 건드리지 않음 (살려둠)
        if log_queue: 
            log_queue.put("✅ 학습 루프 종료 (그래프는 유지됩니다)")

        # 2. GPU 메모리 정리 (이것만 해도 리소스 확보에 큰 도움이 됩니다)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if log_queue: 
            log_queue.put("🧹 GPU Cache Cleared")

        with g_vars.lock:
            g_vars.GLOBAL_CHANGE = True