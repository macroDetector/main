from pydantic import BaseModel
from typing import List, Optional

class RequestBody(BaseModel):
    id: str
    data : List[dict]

class ResponseBody(BaseModel):
    id: str
    status: int
    analysis_results: List[dict]
    message: Optional[str] = None