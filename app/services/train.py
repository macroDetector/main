import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import traceback
import time


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import app.core.globals as globals
import joblib
# 모델
from app.models.MousePoint import MousePoint
from app.models.TransformerMacroDetector import TransformerMacroAutoencoder, MacroDataset

import app.repostitories.DBController as DBController
import app.repostitories.JsonController as JsonController

from app.services.indicators import indicators_generation
from multiprocessing import Queue

from multiprocessing import Process
from multiprocessing import Event

from app.utilites.make_df_from_points import make_df_from_points
from app.utilites.points_to_features import points_to_features


def train_plot_main(train_queue: Queue):
    import sys
    from app.services.RealTimeMonitor import TrainMonitor 
    from PyQt6.QtCore import QTimer
    
    monitor = TrainMonitor(window_size=1000) # Epoch 수에 맞춰 조절
    
    def update():
        # 큐에 쌓인 모든 데이터를 한 번에 처리하여 딜레이 방지
        while not train_queue.empty():
            try:
                data = train_queue.get_nowait()
                # data format: (avg_train_loss, avg_val_loss)
                monitor.update_view(data[0], data[1])
            except Exception as e:
                print(f"Update error: {e}")
                break
                
    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(100) # 그래프 업데이트 주기 (ms)
    
    sys.exit(monitor.app.exec())

class TrainMode():
    def __init__(self, stop_event=None, log_queue:Queue=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.stop_event=stop_event
        self.seq_len=globals.SEQ_LEN
        self.log_queue:Queue=log_queue

        self.plot_proc = Process(
            target=train_plot_main,
            args=(
                globals.TRAIN_DATA, 
            ),
            daemon=False
        )
        self.plot_proc.start()

    # train
    def train_start(self, train_dataset, val_dataset, batch_size=globals.batch_size, epochs=2000, lr=globals.lr,
                    device=None, model=None, stop_event=None, patience=20, log_queue=None, save_path="best_model.pth"):
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 자기지도 학습용: MSELoss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        epochs_no_improve = 0

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
                outputs = model(batch_x)           # 입력 시퀀스 재구성
                loss = criterion(outputs, batch_x) # MSE 계산
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item() * batch_x.size(0)

            avg_train_loss = total_train_loss / len(train_dataset)

            # ===== Validation Phase =====
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch_x in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    total_val_loss += loss.item() * batch_x.size(0)

            avg_val_loss = total_val_loss / len(val_dataset)

            # ===== Logging & Early Stopping =====
            status_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"

            if globals.TRAIN_DATA is not None:
                globals.TRAIN_DATA.put((float(avg_train_loss), float(avg_val_loss)))

            if log_queue: log_queue.put(status_msg)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
                if log_queue: log_queue.put(f"  >> [Model Saved] Best Val Loss: {best_val_loss:.6f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if log_queue: log_queue.put(f"Early Stopping: {patience} epoch 동안 개선 없음.")
                    if stop_event: stop_event.set()
                    break

        if log_queue: log_queue.put(f"학습 종료. 최종 Best Val Loss: {best_val_loss:.6f}")
        
    def clip_outliers(self, df, columns):
        df_clipped = df.copy()
        for col in columns:
            # 상위 1%와 하위 1% 지점 계산
            lower_bound = df_clipped[col].quantile(0.01)
            upper_bound = df_clipped[col].quantile(0.99)
            # 해당 범위를 벗어나는 값들을 경계값으로 고정
            df_clipped[col] = df_clipped[col].clip(lower=lower_bound, upper=upper_bound)
        return df_clipped
        
    def main(self):
        self.log_queue.put(f"device : {self.device} | SEQ_LEN : {globals.SEQ_LEN} | STRIDE : {globals.STRIDE}")

        # ===== 데이터 읽기 =====
        if globals.Recorder == "postgres":
            user_all: list[MousePoint] = DBController.read(True, log_queue=self.log_queue)
            is_dict = False
        elif globals.Recorder == "json":
            user_all: list[dict] = JsonController.read(True, log_queue=self.log_queue)
            is_dict = True

        self.log_queue.put(f"user_all length : {len(user_all)}")

        user_df_chunk = make_df_from_points(user_all, is_dict=is_dict)

        # ===== Feature 계산 =====
        setting_user_df_chunk: pd.DataFrame = indicators_generation(user_df_chunk)
        
        # ===== Feature 필터 =====
        setting_user_df_chunk = setting_user_df_chunk[globals.FEATURES].copy()

        setting_user_df_chunk = self.clip_outliers(setting_user_df_chunk, globals.FEATURES)

        print(f"setting_user_df_chunk : {setting_user_df_chunk}")


        if len(setting_user_df_chunk) < globals.SEQ_LEN:
            self.log_queue.put("데이터가 충분하지 않습니다.")
            return

        # ===== Train/Val split =====
        user_train_df: pd.DataFrame
        user_val_df: pd.DataFrame

        user_train_df, user_val_df = train_test_split(setting_user_df_chunk, test_size=0.2, shuffle=False)

        # ===== Sequence =====q
        user_train_seq, user_train_pass_seq = points_to_features(df_chunk=user_train_df, seq_len=globals.SEQ_LEN, stride=globals.STRIDE, log_queue=self.log_queue)

        user_val_seq, user_val_pass_seq = points_to_features(df_chunk=user_val_df, seq_len=globals.SEQ_LEN, stride=globals.STRIDE, log_queue=self.log_queue)

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
        joblib.dump(scaler, globals.scaler_path)
        
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
            input_size=len(globals.FEATURES),
            d_model=globals.d_model,
            nhead=4,
            num_layers=globals.num_layers,
            dim_feedforward=128,
            dropout=globals.dropout
        ).to(self.device)

        timeinterval = 10

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
            batch_size=globals.batch_size, 
            lr=globals.lr,
            epochs=50,
            device=self.device, 
            model=model, 
            stop_event=self.stop_event, 
            patience=20, 
            log_queue=self.log_queue, 
            save_path=globals.save_path
        )

    def train_stop_event(self, log_queue:Queue=None):
        log_queue.put("🛑 Train Stopped")