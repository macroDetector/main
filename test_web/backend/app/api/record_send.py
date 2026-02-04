import socket
import json
from fastapi import APIRouter, Request
from typing import List
from pydantic import BaseModel
from datetime import datetime
import app.core.globals as g_vars

from app.models.MouseDetectorSocket import ResponseBody, RequestBody

router = APIRouter()

class Pointer(BaseModel):
    timestamp: datetime
    x: int
    y: int
    deltatime: float

@router.post("/get_points")
async def get_mouse_pointer(request: Request, data: List[Pointer]):
    output_list = []
    for p in data:
        p_dict = p.model_dump()
        p_dict['timestamp'] = p.timestamp.replace(tzinfo=None).isoformat()
        output_list.append(p_dict)
    
    async with g_vars.socket_lock:
        sock = None 
        try:
            client_ip = request.client.host
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(60.0) 
            sock.connect(("localhost", 52341))
            
            send_data = RequestBody(
                id=client_ip,
                data=output_list,
            )
            
            final_payload = send_data.model_dump_json().encode('utf-8')

            sock.sendall(final_payload)
            raw_res = sock.recv(65536)

            if not raw_res:
                raise ConnectionError("서버로부터 응답이 없습니다.")
            
            # 5. 응답 역직렬화 및 객체 변환
            res_dict = json.loads(raw_res.decode('utf-8'))
            res_data = ResponseBody(**res_dict)

            if res_data.status != 0:
                return {"status": "error", "message": "Server logic error"}

            print(f"📩 서버 응답 분석 결과: {res_data.analysis_results}")
            return {"status": "success", "results": res_data.analysis_results}
        
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            if sock:
                sock.close()