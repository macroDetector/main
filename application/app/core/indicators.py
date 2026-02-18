import pandas as pd
import numpy as np
from numba import njit
from sklearn.preprocessing import RobustScaler

# 최종 FEATURES
FEATURES_indi = [
    "speed_skew", "acc_skew", "micro_shake_skew", "angle_vel_skew", "straightness_skew",
    "speed_rough", "acc_rough", "micro_shake_rough", "angle_vel_rough", "straightness_rough",
    "speed_tail", "acc_tail", "micro_shake_tail", "angle_vel_tail", "straightness_tail",
    "path_sinuosity", "bending_energy",
]

@njit
def fast_tail_mean(x, q=0.05):
    """Numba 가속: 상하위 q% 평균 (극단값 지표)"""
    n = len(x)
    if n == 0: return 0.0
    sorted_x = np.sort(x)
    k = max(1, int(n * q))
    tail_sum = np.sum(sorted_x[:k]) + np.sum(sorted_x[-k:])
    return tail_sum / (2 * k)

@njit
def fast_entropy(x):
    if np.all(x == x[0]): return 0.0
    # 간단한 빈도수 기반 엔트로피
    unique_vals = np.unique(x)
    probs = np.zeros(len(unique_vals))
    for i, val in enumerate(unique_vals):
        probs[i] = np.sum(x == val) / len(x)
    return -np.sum(probs * np.log2(probs + 1e-9))

def indicators_generation(df_chunk: pd.DataFrame, chunk_size: int = 30, offset: int = 0) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=FEATURES_indi)
    if len(df_chunk) <= chunk_size:
        return empty_df    

    try:
        df = df_chunk.copy()
        eps = 1e-7

        # [A] 기본 물리량 계산
        dt = df["deltatime"].clip(0.005, 0.1)
        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()
        df["dist"] = np.hypot(df["dx"], df["dy"])
        df["speed"] = (df["dist"] / dt).clip(0, 5000)
        df["acc"] = (df["speed"].diff() / dt).clip(-100000, 100000)
        df["theta"] = np.arctan2(df["dy"], df["dx"])
        df["angle_vel"] = np.arctan2(np.sin(df["theta"].diff()), np.cos(df["theta"].diff())) / dt
        df["micro_shake"] = (df["speed"].diff().abs() + df["angle_vel"].diff().abs())

        # [B] 직선도 및 물리량 기반 지표
        total_path_dist = df["dist"].rolling(chunk_size).sum()
        displacement = np.hypot(df["x"] - df["x"].shift(chunk_size), df["y"] - df["y"].shift(chunk_size))
        df["straightness"] = (displacement / (total_path_dist + eps)).clip(0, 1)
        df["path_sinuosity"] = (total_path_dist / (displacement + eps)).clip(0, 100)
        df["bending_energy"] = (df["angle_vel"]**2).rolling(chunk_size).mean()

        # [C] 루프 최적화 (Skew, Rough, Tail)
        target_cols = ["speed", "acc", "straightness", "angle_vel", "micro_shake"]
        total_steps = len(target_cols) * 3
        current_step = 0

        for col in target_cols:
            roll = df[col].rolling(chunk_size)
            
            # 1. Skew
            if f"{col}_skew" in FEATURES_indi:
                df[f"{col}_skew"] = roll.skew()
            current_step += 1
            
            # 2. Rough
            if f"{col}_rough" in FEATURES_indi:
                df[f"{col}_rough"] = df[col].diff().abs().rolling(chunk_size).mean()
            current_step += 1
            
            # 3. Tail (Numba 가속)
            if f"{col}_tail" in FEATURES_indi:
                df[f"{col}_tail"] = roll.apply(fast_tail_mean, raw=True, engine='numba')
            current_step += 1
            
            # 실시간 진행바
            print(f"\rIndicators Progress: [{'■' * current_step}{'□' * (total_steps - current_step)}] {current_step}/{total_steps}", end="", flush=True)

        print("\n[SYSTEM] Post-processing features...")

        # [D] 극단값 처리를 위한 로그 변환 (중요!)
        # 데이터 통계치에서 확인된 폭발적인 값들을 압축합니다.
        log_cols = ["speed_rough", "acc_rough", "micro_shake_rough", "speed_tail", 
                    "micro_shake_tail", "path_sinuosity", "bending_energy"]
        for c in log_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].abs())

        # [E] 마무리 (결측치 및 무한대 제거)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES_indi).reset_index(drop=True)
        if offset > 0:
            df = df.iloc[offset:].reset_index(drop=True)

        return df[FEATURES_indi]

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return empty_df

# [F] 스케일링 및 클리핑 함수 (학습 직전 적용용)
def final_scaling(train_df):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(train_df)
    
    # 스케일링 후에도 남은 극단값을 -10 ~ 10 사이로 제한 (안정성 확보)
    scaled_data = np.clip(scaled_data, -10, 10)
    
    return pd.DataFrame(scaled_data, columns=train_df.columns), scaler