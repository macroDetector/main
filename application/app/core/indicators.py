import pandas as pd
import numpy as np
from numba import njit

FEATURES_indi = [
    # 평균, 표준 편차 => 그래프 형상
    "speed_mean", "speed_std", 
    "acc_mean", "acc_std", 
    "micro_shake_mean", "micro_shake_std", 
    "angle_vel_mean", "angle_vel_std",
    "straightness_mean", "straightness_std",

    # 왜곡, 거칠기 => 그래프의 비대칭성 및 불규칙성
    "speed_skew", "acc_skew", "micro_shake_skew", "angle_vel_skew",
    "speed_rough", "acc_rough", "micro_shake_rough", "angle_vel_rough",
    "straightness_skew", "straightness_rough",

    # 기록기 검거 지표 (무질서도 및 고유값 비율)
    "path_sinuosity", "bending_energy",
]

@njit
def fast_entropy(x):
    """데이터의 무질서도 계산 (기록기의 고정된 타이밍 검거)"""
    if np.all(x == x[0]): return 0.0
    unique_elements = np.unique(x)
    counts = np.zeros(len(unique_elements))
    for i, val in enumerate(unique_elements):
        counts[i] = np.sum(x == val)
    probs = counts / len(x)
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    return entropy

@njit
def fast_unique_ratio(x):
    """고유값 비율 (기록기의 양자화된 좌표/고정값 탐지)"""
    return len(np.unique(x)) / len(x)

def indicators_generation(df_chunk: pd.DataFrame, chunk_size: int = None, offset: int = 0) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=FEATURES_indi)

    if len(df_chunk) <= chunk_size:
        return empty_df    
    try:
        df = df_chunk.copy()
        eps = 1e-7

        # [A] 기본 물리량 (안정성 강화)
        dt = df["deltatime"].clip(0.005, 0.1) 
        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()
        df["dist"] = np.hypot(df["dx"], df["dy"])
        
        df["speed"] = (df["dist"] / dt).clip(0, 5000)
        df["acc"] = (df["speed"].diff() / dt).clip(-100000, 100000)
        
        # 방향 전환 및 미세 떨림
        df["theta"] = np.arctan2(df["dy"], df["dx"])
        df["angle_vel"] = np.arctan2(np.sin(df["theta"].diff()), np.cos(df["theta"].diff())) / dt
        df["micro_shake"] = (df["speed"].diff().abs() + df["angle_vel"].diff().abs())

        # [B] 직선도(Straightness) 기본값 계산
        # 윈도우 내에서의 실제 변위와 총 이동거리의 비율
        total_path_dist = df["dist"].rolling(chunk_size).sum()
        displacement = np.hypot(
            df["x"] - df["x"].shift(chunk_size), 
            df["y"] - df["y"].shift(chunk_size)
        )
        df["straightness"] = (displacement / (total_path_dist + eps)).clip(0, 1)

        # [C] 통계 지표 생성 루프 (직선도 포함)
        # 이제 직선도(straightness)도 속도나 가속도처럼 통계 분석을 거칩니다.
        target_cols = ["speed", "acc", "straightness", "angle_vel", "micro_shake"]
        for col in target_cols:
            roll = df[col].rolling(chunk_size)
            
            # 가우시안 차트용 Mean, Std
            df[f"{col}_mean"] = roll.mean()
            df[f"{col}_std"] = roll.std()
            
            # 극단값/불규칙성 탐지용 Skew, Rough
            if f"{col}_skew" in FEATURES_indi:
                df[f"{col}_skew"] = roll.skew()
            if f"{col}_rough" in FEATURES_indi:
                df[f"{col}_rough"] = df[col].diff().abs().rolling(chunk_size).mean()

        # [D] 기타 핵심 분석
        df["dt_entropy"] = df["deltatime"].rolling(chunk_size).apply(fast_entropy, raw=True, engine='numba')
        df["speed_entropy"] = df["speed"].rolling(chunk_size).apply(fast_entropy, raw=True, engine='numba')
        df["bending_energy"] = (df["angle_vel"]**2).rolling(chunk_size).mean()
        df["path_sinuosity"] = (total_path_dist / (displacement + eps)).clip(0, 100)

        # [E] 최종 클린업
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if offset > 0:
            df = df.iloc[offset:].reset_index(drop=True)    
        
        return df[FEATURES_indi]
    except Exception as e:
        return empty_df
    