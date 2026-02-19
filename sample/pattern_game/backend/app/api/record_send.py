from fastapi import APIRouter
from typing import List
import os
import json
from datetime import datetime
from QMacroDetector import Pattern_Game, MousePoint
from QMacroDetector.Response import ResponseBody
import numpy as np
from scipy import stats

router = APIRouter()

# SAVE_DIR = os.path.join(os.getcwd(), "data", "captured_movements")
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR, exist_ok=True)

@router.post("/get_points")
async def get_mouse_pointer(data: List[MousePoint]):   
    # try:
    #     json_ready_data = [
    #         {
    #             "timestamp": p.timestamp.isoformat(),
    #             "x": p.x,
    #             "y": p.y,
    #             "deltatime": p.deltatime
    #         } 
    #         for p in data
    #     ]

    #     timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    #     file_path = os.path.join(SAVE_DIR, f"move_{timestamp_str}.json")

    #     with open(file_path, "w", encoding="utf-8") as f:
    #         json.dump(json_ready_data, f, ensure_ascii=False, indent=4)
            
    # except Exception as e:
    #     print(f"❌ 데이터 저장 실패: {e}")

    # print(data)
    result:ResponseBody = Pattern_Game().get_macro_result(data)

    print(result)
    received_data:list = result.data

    # print(received_data)
    
    error_mean = stats.trim_mean(received_data, proportiontocut=0.05)
    
    # print(error_mean)
    return {
        "status" : 0,
        "message" : float(error_mean)
    }