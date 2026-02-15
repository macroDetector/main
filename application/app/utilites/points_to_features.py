import app.core.globals as g_vars
import numpy as np
import pandas as pd
from multiprocessing import Queue
import traceback

# stride = > seq 데이터 겹치는 양 (How much each sequence shifts forward)
def points_to_features(df_chunk: pd.DataFrame, seq_len: int = g_vars.SEQ_LEN, stride: int = g_vars.STRIDE, log_queue:Queue=None):
    Total_Pass_SEQ = 0
    
    # 시퀀스내의 스탭들이 모드 동일할 때 제거 => 움직임이 없을 경우 제외
    def df_to_seq(df: pd.DataFrame):
        nonlocal Total_Pass_SEQ
        try:
            X = []
            n_rows = len(df)

            for i in range(0, n_rows - seq_len + 1, stride):
                seq = df.iloc[i:i + seq_len][g_vars.FEATURES].values

                if np.all(seq==0):
                    continue

                Total_Pass_SEQ += 1
                X.append(seq)

            if len(X) == 0:
                return np.empty((0, seq_len, len(g_vars.FEATURES)), dtype=np.float32)
            return np.asarray(X, dtype=np.float32)
        
        except Exception as e:
            log_queue.put("Error occurred in df_to_seq:")
            log_queue.put(e)                   
            traceback.print_exc()      
            return np.empty((0, seq_len, len(g_vars.FEATURES)), dtype=np.float32)

    data = df_to_seq(df_chunk)

    return data, Total_Pass_SEQ