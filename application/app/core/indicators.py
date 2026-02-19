import pandas as pd
import numpy as np
from numba import njit
from sklearn.preprocessing import RobustScaler

FEATURES_indi = [
    "speed_skew", "acc_skew", "micro_shake_skew", "angle_vel_skew", "straightness_skew",
    "speed_rough", "acc_rough", "micro_shake_rough", "angle_vel_rough", "straightness_rough",
    "speed_tail", "acc_tail", "micro_shake_tail", "angle_vel_tail", "straightness_tail",
    "path_sinuosity", "bending_energy",

    # 🔥 NEW — macro detection
    "linear_ratio", "max_linear_run", "linear_run_mean",
    "curvature_std",
    "jerk_energy", "jerk_std",
    "submovement_count",
    "dt_std"
]

@njit
def fast_tail_mean(x, q=0.05):
    n = len(x)
    if n == 0:
        return 0.0
    sorted_x = np.sort(x)
    k = max(1, int(n * q))
    tail_sum = np.sum(sorted_x[:k]) + np.sum(sorted_x[-k:])
    return tail_sum / (2 * k)


@njit
def linear_metrics(theta, speed, acc, angle_thresh=0.01, acc_thresh=500):
    """
    직선 구간 분석
    """
    n = len(theta)
    if n < 3:
        return 0.0, 0.0, 0.0

    linear_mask = np.zeros(n - 1)

    for i in range(1, n):
        dtheta = abs(theta[i] - theta[i - 1])
        dacc = abs(acc[i] - acc[i - 1])

        if dtheta < angle_thresh and dacc < acc_thresh:
            linear_mask[i - 1] = 1

    # ratio
    linear_ratio = np.mean(linear_mask)

    # run length
    runs = []
    run = 0

    for v in linear_mask:
        if v == 1:
            run += 1
        else:
            if run > 0:
                runs.append(run)
            run = 0

    if run > 0:
        runs.append(run)

    if len(runs) == 0:
        return linear_ratio, 0.0, 0.0

    max_run = np.max(np.array(runs))
    mean_run = np.mean(np.array(runs))

    return linear_ratio, max_run, mean_run


def indicators_generation(df_chunk: pd.DataFrame, chunk_size: int = 30, offset: int = 0) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=FEATURES_indi)

    if len(df_chunk) <= chunk_size:
        return empty_df

    try:
        df = df_chunk.copy()
        eps = 1e-7

        dt = df["deltatime"].clip(0.005, 0.1)

        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()

        df["dist"] = np.hypot(df["dx"], df["dy"])
        df["speed"] = (df["dist"] / dt).clip(0, 5000)
        df["acc"] = (df["speed"].diff() / dt).clip(-100000, 100000)

        df["theta"] = np.arctan2(df["dy"], df["dx"])
        df["angle_vel"] = np.arctan2(np.sin(df["theta"].diff()), np.cos(df["theta"].diff())) / dt

        df["micro_shake"] = (df["speed"].diff().abs() + df["angle_vel"].diff().abs())

        # curvature
        df["curvature"] = df["angle_vel"] / (df["speed"] + eps)

        # jerk
        df["jerk"] = df["acc"].diff() / dt

        # straightness
        total_path_dist = df["dist"].rolling(chunk_size).sum()
        displacement = np.hypot(
            df["x"] - df["x"].shift(chunk_size),
            df["y"] - df["y"].shift(chunk_size)
        )

        df["straightness"] = (displacement / (total_path_dist + eps)).clip(0, 1)
        df["path_sinuosity"] = (total_path_dist / (displacement + eps)).clip(0, 100)
        df["bending_energy"] = (df["angle_vel"]**2).rolling(chunk_size).mean()

        # 기존 통계 features
        target_cols = ["speed", "acc", "straightness", "angle_vel", "micro_shake"]

        for col in target_cols:
            roll = df[col].rolling(chunk_size)

            df[f"{col}_skew"] = roll.skew()
            df[f"{col}_rough"] = df[col].diff().abs().rolling(chunk_size).mean()
            df[f"{col}_tail"] = roll.apply(fast_tail_mean, raw=True, engine='numba')

        # 🔥 NEW FEATURES 계산
        linear_ratio_list = []
        max_run_list = []
        mean_run_list = []
        curvature_std_list = []
        jerk_energy_list = []
        jerk_std_list = []
        submove_list = []
        dt_std_list = []

        for i in range(chunk_size, len(df)):
            chunk = df.iloc[i - chunk_size:i]

            theta = chunk["theta"].values
            speed = chunk["speed"].values
            acc = chunk["acc"].values

            lr, mr, meanr = linear_metrics(theta, speed, acc)

            linear_ratio_list.append(lr)
            max_run_list.append(mr)
            mean_run_list.append(meanr)

            curvature_std_list.append(np.std(chunk["curvature"]))
            jerk_energy_list.append(np.mean(chunk["jerk"]**2))
            jerk_std_list.append(np.std(chunk["jerk"]))

            # submovement count (speed local minima)
            sp = chunk["speed"].values
            minima = np.sum((sp[1:-1] < sp[:-2]) & (sp[1:-1] < sp[2:]))
            submove_list.append(minima)

            dt_std_list.append(np.std(chunk["deltatime"]))

        pad = [np.nan] * chunk_size

        df["linear_ratio"] = pad + linear_ratio_list
        df["max_linear_run"] = pad + max_run_list
        df["linear_run_mean"] = pad + mean_run_list
        df["curvature_std"] = pad + curvature_std_list
        df["jerk_energy"] = pad + jerk_energy_list
        df["jerk_std"] = pad + jerk_std_list
        df["submovement_count"] = pad + submove_list
        df["dt_std"] = pad + dt_std_list

        # 로그 압축
        log_cols = [
            "speed_rough", "acc_rough", "micro_shake_rough",
            "speed_tail", "micro_shake_tail",
            "path_sinuosity", "bending_energy",
            "jerk_energy"
        ]

        for c in log_cols:
            if c in df.columns:
                df[c] = np.log1p(np.abs(df[c]))

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=FEATURES_indi).reset_index(drop=True)

        if offset > 0:
            df = df.iloc[offset:].reset_index(drop=True)

        return df[FEATURES_indi]

    except Exception as e:
        print(f"[ERROR] {e}")
        return empty_df


def final_scaling(train_df):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(train_df)
    scaled_data = np.clip(scaled_data, -10, 10)

    return pd.DataFrame(scaled_data, columns=train_df.columns), scaler
