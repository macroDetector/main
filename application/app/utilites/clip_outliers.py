import pandas as pd

def clip_outliers(df:pd.DataFrame, columns):
    df_clipped = df.copy()
    for col in columns:
        # 상위 1%와 하위 1% 지점 계산
        lower_bound = df_clipped[col].quantile(0.01)
        upper_bound = df_clipped[col].quantile(0.99)
        # 해당 범위를 벗어나는 값들을 경계값으로 고정
        df_clipped[col] = df_clipped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_clipped