import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()
    
    # 위치 변화량 & 거리
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    
    # 속도 및 로그 속도
    df["speed"] = df["dist"] / df["deltatime"]
    
    # 가속도 & 변화량
    df["acc"] = df["speed"].diff()

    # jerk & 변화량
    df["jerk"] = df["acc"].diff()

    # 이동 각도, 방향 변화, 방향 가속도
    df["angle"] = np.arctan2(df["dy"], df["dx"])
    df["turn"] = df["angle"].diff()
    df["turn"] = (df["turn"] + np.pi) % (2 * np.pi) - np.pi  # wrap
    df["turn_acc"] = df["turn"].diff()
    
    
    # NaN/inf → 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df
