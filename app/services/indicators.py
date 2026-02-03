import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()
    
    # 0. 시간 안전장치 (dt가 너무 작으면 속도가 폭발함)
    dt = df["deltatime"].replace(0, 0.0001)
    
    df["dt_diff"] = df["deltatime"].diff().abs()
    df["dt_cv"] = (
        df["deltatime"].rolling(10, min_periods=1).std() /
        (df["deltatime"].rolling(10, min_periods=1).mean() + 1e-6)
    )

    # 1. 기본 물리량 계산 (이동 거리, 속도, 가속도, 저크)
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    
    df["speed"] = df["dist"] / dt
    df["acc"] = df["speed"].diff()
    df["jerk"] = df["acc"].diff()

    # 2. [추가] 미세 떨림 (Micro-Jitter) 분석
    # 사람이 이동 중 발생하는 불규칙한 미세 움직임 포착
    df["jitter_x"] = df["dx"].diff().abs()
    df["jitter_y"] = df["dy"].diff().abs()
    df["micro_shaking"] = (df["jitter_x"] + df["jitter_y"]).rolling(5, min_periods=1).mean()

    # 3. 가속도 방향 전환 빈도 (Jerk Flip)
    df["jerk_sign"] = np.sign(df["jerk"])
    df["jerk_flip"] = (df["jerk_sign"].diff().abs() > 1).astype(int)
    df["jerk_flip_rate"] = df["jerk_flip"].rolling(10, min_periods=1).mean()

    # 4. 방향 및 회전 관련 (Turn, Angular Velocity)
    df["angle"] = np.arctan2(df["dy"], df["dx"])
    df["turn"] = (df["angle"].diff() + np.pi) % (2 * np.pi) - np.pi
    
    # [추가] 각속도 및 각가속도 (곡선의 부드러움 측정)
    df["ang_vel"] = df["turn"] / dt
    df["ang_acc"] = df["ang_vel"].diff()

    # 5. 통계 피처 (분산 및 부드러움)
    df["speed_var"] = df["speed"].rolling(window=10, min_periods=1).std()
    
    # [추가] 가속도의 일정함 (매크로는 가속도 변화가 매우 규칙적임)
    df["acc_smoothness"] = df["acc"].abs() / (df["acc"].rolling(10, min_periods=1).std() + 1e-6)

    # 6. 직선성 (Straightness)
    window_s = 20
    rolling_dist_sum = df["dist"].rolling(window=window_s).sum()
    line_dist = np.sqrt(
        (df["x"] - df["x"].shift(window_s-1))**2 + 
        (df["y"] - df["y"].shift(window_s-1))**2
    ).clip(lower=1e-6)
    df["straightness"] = (rolling_dist_sum / line_dist).fillna(1.0)
    df["straightness"] = df["straightness"].clip(1, 5)

    # 7. [추가] 경로 효율의 변화량
    # 사람이 목표물을 향해 갈 때 발생하는 미세 보정 탐지
    df["efficiency_var"] = df["straightness"].diff().abs().rolling(10, min_periods=1).mean()

    # 8. NaN/inf 최종 정리
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df