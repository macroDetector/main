import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import pandas as pd
import time
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import app.core.globals as g_vars
import joblib

from app.models.TransformerMacroDetector import TransformerMacroAutoencoder, MacroDataset

import app.repostitories.JsonController as JsonController

from app.core.indicators import indicators_generation
from multiprocessing import Queue

from app.utilites.make_df_from_points import make_df_from_points
from app.utilites.make_sequence import make_seq

from app.utilites.save_confing import update_parameters
from app.utilites.loss_caculation import Loss_Calculation

def train_plot_main(train_queue: Queue):
    import sys
    from app.utilites.train_plot_monitor import TrainMonitor
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

    def train_start(self, train_dataset, val_dataset, batch_size=g_vars.batch_size, epochs=2000, lr=g_vars.lr, 
                    device=None, model=None, stop_event=None, patience=20, log_queue:Queue=None, save_path="best_model.pth"):
        try:
            log_queue.put(f"🚀 학습 프로세스 시작 | LR: {lr}")
            self.start_plot_process()

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            criterion = criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=g_vars.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
            )

            best_val_loss = float('inf')
            epochs_no_improve = 0
            standard_val_loss = None 
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

                        sample_errors = Loss_Calculation(outputs=outputs, batch=batch_x)
                        
                        all_val_errors.extend(sample_errors.cpu().numpy())
                        
                        loss = criterion(outputs, batch_x)
                        total_val_loss += loss.item() * batch_x.size(0)

                avg_val_loss = total_val_loss / len(val_dataset)
                
                # 차트 데이터 전송 (이 부분이 있어야 실시간 그래프가 그려집니다)
                if g_vars.TRAIN_DATA:
                    g_vars.TRAIN_DATA.put((epoch + 1, float(avg_train_loss), float(avg_val_loss)))
                
                # 개선율 기준점 지연 설정 (Epoch 3)
                if standard_val_loss is None and epoch >= 2: 
                    standard_val_loss = avg_val_loss
                    log_queue.put(f"📍 기준점 설정(E{epoch+1}): {standard_val_loss:.6f}")
                
                scheduler.step(avg_val_loss)

                errors = np.array(all_val_errors)

                # 1. 중앙값과 MAD 계산 (이상치에 절대 안 휘둘림)
                median_e = np.median(errors)
                mad_e = np.median(np.abs(errors - median_e))

                # 2. 통계적으로 '정상 범위' 밖의 에러들(Outliers)을 완전히 제외하고 다시 계산
                # 보통 Median + 3 * (1.4826 * MAD) 밖의 값은 쓰레기 데이터로 봅니다.
                upper_limit = median_e + (3 * 1.4826 * mad_e)
                refined_errors = errors[errors < upper_limit]

                # 3. 정제된 에러들로만 다시 평균, 표준편차 계산
                m_e = np.mean(refined_errors)
                s_e = np.std(refined_errors)

                # 4. 여기서 배수(Multiplier) 5.0~6.0 적용
                multiplier = 6.0 
                current_epoch_threshold = m_e + (multiplier * s_e)

                # 5. 안전장치: 현재 Val(평균)보다는 커야 함
                current_epoch_threshold = max(current_epoch_threshold, np.mean(errors) * 1.2)

                # 로그 출력
                status_msg = f"Epoch {epoch+1} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Thres: {current_epoch_threshold:.6f}"
                if log_queue: log_queue.put(status_msg)

                # 개선율 체크 및 조기 종료
                if standard_val_loss is not None:
                    improvement_total = (standard_val_loss - avg_val_loss) / standard_val_loss
                    if log_queue: log_queue.put(f"📊 개선율: {improvement_total * 100:.2f}% / 목표: {g_vars.improvement_val_loss_cut * 100}%")
                    
                    if improvement_total >= g_vars.improvement_val_loss_cut:
                        recommended_base_threshold = current_epoch_threshold
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), save_path)
                        g_vars.threshold = recommended_base_threshold 
                        update_parameters({"THRES" : recommended_base_threshold})
                        with g_vars.lock:
                            g_vars.GLOBAL_CHANGE = True                                                   
                        if log_queue: log_queue.put(f"🎯 목표 달성! ({improvement_total*100:.1f}%) 학습을 종료합니다.")
                        break

                # Best 모델 저장 로직
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    recommended_base_threshold = current_epoch_threshold
                    torch.save(model.state_dict(), save_path)
                    g_vars.threshold = recommended_base_threshold 
                    update_parameters({"THRES" : recommended_base_threshold})  
                    with g_vars.lock:
                        g_vars.GLOBAL_CHANGE = True                                      
                    epochs_no_improve = 0
                    if log_queue: log_queue.put(f" >> [Best Saved] Loss: {best_val_loss:.6f}")
                else:
                    epochs_no_improve += 1
                    if log_queue: log_queue.put(f" >> [No Improve] {epochs_no_improve}/{patience}")
                    if epochs_no_improve >= patience:
                        if log_queue: log_queue.put(f"Early Stopping 발생!")
                        break
        finally:
            self.train_stop_event(log_queue=log_queue)
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def main(self):
        self.log_queue.put(f"device : {self.device} | SEQ_LEN : {g_vars.SEQ_LEN} | STRIDE : {g_vars.STRIDE}")

        user_all: list[dict]
        filename, user_all = JsonController.read(log_queue=self.log_queue)
        is_dict = True
            
        user_df_chunk = make_df_from_points(user_all, is_dict=is_dict)

        user_df_chunk= user_df_chunk.sort_values('timestamp').reset_index(drop=True)

        user_df_chunk = user_df_chunk[user_df_chunk["deltatime"] <= g_vars.filter_tolerance].reset_index(drop=True)

        pd.options.display.float_format = '{:,.4f}'.format
        def print_box(title, content, color_code="36"): # 36: Cyan, 32: Green, 33: Yellow
            width = 60
            print(f"\033[{color_code}m" + "="*width)
            print(f"  {title.center(width-4)}")
            print("="*width + "\033[0m")
            print(content)
            print("\033[" + color_code + "m" + "-"*width + "\033[0m\n")

        # ===== 지표 생성 ======
        setting_user_df_chunk: pd.DataFrame = indicators_generation(
            df_chunk=user_df_chunk, 
            chunk_size=g_vars.chunk_size, 
            offset=g_vars.offset
        )

        setting_user_df_chunk = setting_user_df_chunk[g_vars.FEATURES].copy()

        stats_before = setting_user_df_chunk[g_vars.FEATURES].agg(['min', 'max', 'mean', 'std']).T
        print_box("📊 RAW DATA STATISTICS (Before Scaling)", stats_before, "33")        
        print("✅ 지표 생성 성공")
        
        # ==== 스케일 작업 =====
        scaler = MinMaxScaler()

        scaled_array = scaler.fit_transform(setting_user_df_chunk[g_vars.FEATURES])
        scaled_array = scaled_array * 10 # 테스트 10배
        chunks_scaled_df = pd.DataFrame(scaled_array, columns=g_vars.FEATURES)

        base_dir = g_vars.scaler_path
        final_save_path = os.path.join(base_dir, f"{filename}_scaler.pkl")
        stats_after = chunks_scaled_df.agg(['min', 'max', 'mean', 'std']).T
        print_box("🚀 SCALED DATA STATISTICS ", stats_after, "32")
        joblib.dump(scaler, final_save_path)
        print(f"✅ Cliping Save")
        
        final_input:np.array = make_seq(data=chunks_scaled_df, seq_len=g_vars.SEQ_LEN, stride=g_vars.STRIDE)

        print(f"✅ 최종 시퀀스 Shape: {final_input.shape}")
        print(f"🚀 첫 번째 시퀀스 내부 덩어리 예시:\n{final_input[0][0]}")

        # ==== 데이터 셋 정의 ====
        train, val = train_test_split(final_input, test_size=0.2, shuffle=True)

        train_dataset = MacroDataset(train)
        val_dataset   = MacroDataset(val)

        # ==== 모델 정의 ====
        model = TransformerMacroAutoencoder(
            input_size=g_vars.input_size,
            d_model=g_vars.d_model,
            nhead=g_vars.n_head,
            num_layers=g_vars.num_layers,
            dim_feedforward=g_vars.dim_feedforward,
            dropout=g_vars.dropout
        ).to(self.device)

        # ==== 시작 타이머 ====
        timeinterval = 5

        while timeinterval != 0:
            if self.stop_event.is_set():
                self.train_stop_event(log_queue=self.log_queue)
                return

            timeinterval -= 1
            self.log_queue.put(f"train 시작까지 count : {timeinterval}")

            time.sleep(1)

        base_dir = g_vars.save_path
        final_save_path = os.path.join(base_dir, f"{filename}_model.pt")
        
        # ==== 트레인 정의 ====
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
            save_path=final_save_path,
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