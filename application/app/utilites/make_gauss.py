import numpy as np
import sys
import pandas as pd
from numpy.lib.stride_tricks import as_strided

def make_gauss(data: pd.DataFrame, chunk_size: int, chunk_stride: int, offset: int, train_mode: bool = True) -> np.array:
    data_np = data.values[offset:].astype(np.float64)
    n_samples, n_features = data_np.shape
    eps = 1e-9

    num_chunks = (n_samples - chunk_size) // chunk_stride + 1
    if num_chunks <= 0: return np.array([])
    
    itemsize = data_np.itemsize
    chunks = as_strided(
        data_np,
        shape=(num_chunks, chunk_size, n_features),
        strides=(chunk_stride * n_features * itemsize, n_features * itemsize, itemsize)
    )

    # 1. 기본 통계량
    m = np.mean(chunks, axis=1, keepdims=True)
    diff = chunks - m
    s = np.std(chunks, axis=1, ddof=0)
    s_safe = s + eps

    # 지표 1: 왜도 (대칭성)
    sk = np.mean(diff**3, axis=1) / (s_safe**3)
    
    # 지표 2: 거칠기 (1차 미분 - 속도의 변화)
    roughness = np.mean(np.abs(np.diff(chunks, axis=1)), axis=1)

    # 지표 3: 가속도 거칠기 (2차 미분 - 기록기 보간법 검거용)
    # 유저는 떨림 때문에 이게 크고, 매크로는 계산된 부드러운 곡선이라 이게 매우 작습니다.
    jerk_rough = np.mean(np.abs(np.diff(chunks, n=2, axis=1)), axis=1)

    # 2. 엔트로피 및 고유값 분석
    actual_entropy = np.zeros((num_chunks, n_features))
    unique_ratio = np.zeros((num_chunks, n_features)) # 지표 4: 고유값 비율
    
    for idx in range(num_chunks):
        window = chunks[idx]
        for col in range(n_features):
            col_data = window[:, col]
            
            # 실측 엔트로피
            counts, _ = np.histogram(col_data, bins=10)
            p = counts / (counts.sum() + eps)
            p = p[p > 0]
            actual_entropy[idx, col] = -np.sum(p * np.log2(p))
            
            # 고유값 비율: 기록기는 유저보다 똑같은 값이 반복될 확률이 높음
            unique_ratio[idx, col] = len(np.unique(col_data)) / chunk_size

        if train_mode and ((idx + 1) % max(1, (num_chunks // 50)) == 0 or (idx + 1) == num_chunks):
            progress = (idx + 1) / num_chunks
            bar = '■' * int(20 * progress) + '□' * (20 - int(20 * progress))
            sys.stdout.write(f'\r특징 추출 중: [{bar}] {progress*100:>5.1f}% ({idx+1}/{num_chunks})')
            sys.stdout.flush()

    # 이론적 엔트로피 및 갭
    theo_entropy = 0.5 * np.log2(2 * np.pi * np.e * (s_safe**2) + eps)
    entropy_gap = theo_entropy - actual_entropy
    
    # [수정] 지표별로 5가지 특성을 묶음 (Sk, Gap, Rough, Jerk, Unique)
    # 이제 한 지표당 5개의 파라미터가 들어갑니다.
    combined = np.stack([sk, entropy_gap, roughness, jerk_rough, unique_ratio], axis=-1)
    result = combined.reshape(num_chunks, -1) 

    if train_mode:
        sys.stdout.write('\n')
    return result