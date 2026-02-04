import time
import app.core.globals as g_vars
from datetime import datetime
from multiprocessing import Queue

from app.services.macro_dectector import MacroDetector
from multiprocessing import Event
from app.repostitories.JsonController import read

def main(stop_event=None, log_queue:Queue=None, chart_Show=True, mode:str = "2"):
    if stop_event is None:
        stop_event = Event()

    detector = MacroDetector(
        model_path=g_vars.save_path,
        seq_len=g_vars.SEQ_LEN,
        threshold=g_vars.threshold,
        chart_Show=chart_Show,
        stop_event=stop_event
    )

    detector.start_plot_process()

    user_data:list[dict]

    # 소켓 모드
    if mode == "1":
        import socket
        import json

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("localhost", 52341))
        server_socket.listen(1)
        
        # 1초마다 대기 상태를 풀고 stop_event를 체크하게 함
        server_socket.settimeout(1.0)

        print("🚀 데이터 수신 대기 중... (중지하려면 stop_event 발생 필요)")
        
        client_socket = None
        while not stop_event.is_set():
            try:
                client_socket, addr = server_socket.accept()
                print(f"✅ 연결됨: {addr}")
                break # 연결 성공 시 대기 루프 탈출
            except socket.timeout:
                # 1초 지남 -> 아무 일도 없었으니 다시 while문 처음으로 가서 stop_event 확인
                continue 
            except Exception as e:
                print(f"❌ 접속 오류: {e}")
                break

        # 만약 연결되지 않고 stop_event가 세팅되어 루프를 빠져나왔다면 종료
        if client_socket is None:
            server_socket.close()
            return []

        try:
            # 클라이언트로부터 데이터 수신
            data = client_socket.recv(1024 * 1024) 
            if not data:
                raise Exception("데이터가 비어있습니다.")
            
            user_data = json.loads(data.decode('utf-8'))
            print(f"📩 수신 완료: {len(user_data)} 건")

            # 수신 확인 응답
            response = {"status": "success", "message": "Ready"}
            client_socket.sendall(json.dumps(response).encode('utf-8'))
        except Exception as e:
            print(f"❌ 데이터 처리 오류: {e}")
            user_data = []
            client_socket.close()
            server_socket.close()
    # json file load
    elif mode == "2":
        import os
        from tkinter import filedialog
        from tkinter import Tk        
        import json

        file_pahh = filedialog.askopenfilename(title="Json 파일을 선택해 주세요", filetypes=(("json 파일", "*.json"), ("모든 파일", "*.*")))
        if not os.path.exists(file_pahh):
            return [] 

        try:
            with open(file_pahh, "r", encoding="utf-8") as f:
                data = json.load(f)
        
            user_data = data
        except Exception as e:
            print(e)
            user_data = []

    timeinterval = 7

    while timeinterval != 0:
        timeinterval -= 1
        if log_queue:
            log_queue.put(f"inference 시작까지 count : {timeinterval}")
        else:
            print(f"inference 시작까지 count : {timeinterval}")

        time.sleep(1)

    if log_queue:
        log_queue.put("🟢 Macro Detector Running")
    else:
        print("🟢 Macro Detector Running")

    print(user_data[:5])

    all_data = []
    for step in user_data:
        if stop_event.is_set():
            log_queue.put("🛑 Detector 중지")
            break
        data = {
            'timestamp': datetime.fromisoformat(step.get("timestamp")),
            'x': step.get("x"),
            'y': step.get("y"),
            'deltatime': step.get("deltatime")  
        }
        result = detector.push(data)

        if result:
            # 확률 수치(float)를 가져옵니다.
            m_prob = result.get('prob_value', 0.0) 
            m_str = result.get('macro_probability', "0%")
            raw_e = result.get('raw_error', 0.0)

            if result["is_human"]:
                log_msg = f"{m_str} (err: {raw_e:.4f})"
            else:
                # 매크로 판정 시 사이렌 이모지와 함께 확률 강조
                log_msg = f"{m_str} (err: {raw_e:.4f}) 🚨"

            # 출력 대상 선택 (Queue 혹은 Print)
            if log_queue:
                log_queue.put(log_msg)
            else:
                print(log_msg)
                if mode == "1":
                    all_data.append(log_msg)

    if mode == "1" and 'client_socket' in locals():
        try:
            # 결과를 JSON으로 묶어 전송
            result_json = json.dumps({"analysis_results": all_data}).encode('utf-8')
            client_socket.sendall(result_json)
            print("📤 모든 분석 결과 전송 완료")
        except Exception as e:
            print(f"❌ 결과 전송 중 오류: {e}")
        finally:
            client_socket.close()
            server_socket.close()
        
    try:
        # stop_event가 발생할 때까지 메인 프로세스는 대기
        while not stop_event.is_set():
            time.sleep(0.1)
    except Exception as e:
        if log_queue:
            log_queue.put(f"에러 발생: {e}")
        else:
            print(f"에러 발생: {e}")
    finally:
        if log_queue:
            log_queue.put("🛑 Detector 종료")
        else:
            print("🛑 Detector 종료")
        stop_event.set()

    if log_queue:
        log_queue.put("🛑 Macro Detector Stopped")
    else:
        print("🛑 Macro Detector Stopped")

    stop_event.set()