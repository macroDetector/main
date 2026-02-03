import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()
    
    # 0. 시간 안전장치 (dt가 너무 작으면 속도가 폭발함)
    dt = df["deltatime"].clip(lower=0.001) + 1e-6
    
    # 1. 기본 물리량 계산
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    
    # 기본 물리량 (이제 로그 안 씌웁니다)
    df["speed"] = df["dist"] / dt
    df["acc"] = df["speed"].diff()
    df["jerk"] = df["acc"].diff()
    # jerk_diff는 정보 중복이 심하므로 생략하거나 유지 (선택사항)
    # df["jerk_diff"] = df["jerk"].diff()

    # 2. 방향 및 회전 관련
    df["angle"] = np.arctan2(df["dy"], df["dx"])
    # 0도 근처에서 360도로 튀는 현상 방지 (Wrap-around)
    df["turn"] = (df["angle"].diff() + np.pi) % (2 * np.pi) - np.pi
    
    # 3. 보간법 탐지용 통계 피처
    # 윈도우 크기를 5 -> 10 정도로 늘리면 매크로의 '일정함'이 더 잘 보입니다.
    df["jerk_std"] = df["jerk"].rolling(window=10, min_periods=1).std()
    df["speed_var"] = df["speed"].rolling(window=10, min_periods=1).std()
    
    # 직선성: 윈도우를 조금 더 길게 가져가면 매크로의 직선 경로가 명확히 잡힙니다.
    window_s = 10
    rolling_dist_sum = df["dist"].rolling(window=window_s).sum()
    line_dist = np.sqrt(
        (df["x"] - df["x"].shift(window_s-1))**2 + 
        (df["y"] - df["y"].shift(window_s-1))**2
    ).clip(lower=1e-6)
    df["straightness"] = (rolling_dist_sum / line_dist).fillna(1.0)

    # -------------------------------------------------------
    # 4. 로그 변환 제거! 대신 Clip으로 이상치만 방어
    # -------------------------------------------------------
    # 로그 없이 쌩 데이터로 보냅니다. 
    # 대신 RobustScaler가 처리하기 힘든 극단적인 값만 잘라줍니다.
    # 이 작업은 이미 MacroDetector의 _infer 단계에서 CLIP_BOUNDS로 하고 있으므로 
    # 여기서는 생략해도 됩니다.

    # 직선성 상한선은 유지 (5.0만 넘어도 이미 엄청나게 휘어지는 것)
    df["straightness"] = df["straightness"].clip(1, 5)

    # 5. NaN/inf 최종 정리
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df