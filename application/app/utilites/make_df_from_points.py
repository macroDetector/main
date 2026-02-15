import pandas as pd
from app.models.MousePoint import MousePoint
from typing import List, Union

def make_df_from_points(points: Union[List[MousePoint], List[dict]], is_dict=False) -> pd.DataFrame:
    if is_dict:
        return pd.DataFrame({
            "timestamp": [p.get("timestamp") for p in points],
            "x": [p.get("x") for p in points],
            "y": [p.get("y") for p in points],
            "deltatime": [p.get("deltatime") for p in points],            
        })
    else:
        return pd.DataFrame({
            "timestamp": [p.timestamp for p in points],
            "x": [p.x for p in points],
            "y": [p.y for p in points],
            "deltatime": [p.deltatime for p in points],                
        })
    