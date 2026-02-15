from fastapi import APIRouter

from QMacroDetector import Pattern_Game, MousePoint

router = APIRouter()

@router.websocket("/get_points_live")
async def get_mouse_pointer_live(data: MousePoint):
    result = Pattern_Game().get_macro_result_live(data)

    print(result)
