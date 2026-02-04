import time
import app.core.globals as g_vars
from datetime import datetime
from multiprocessing import Queue
from collections import deque
from app.services.macro_dectector import MacroDetector
from multiprocessing import Event
from app.models.MouseDetectorSocket import ResponseBody, RequestBody
import socket
import json

# 소켓
# 대기시간 삭제
# while 문으로 무한 유지
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

    if log_queue : log_queue.put(f"weight_threshold : {g_vars.weight_threshold}")
    else:
        print(f"weight_threshold : {g_vars.weight_threshold}")
        
    user_data:list[dict]

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("localhost", 52341))
    server_socket.listen(5)
    server_socket.settimeout(1.0)
    
    print("🚀 서버가 포트 52341에서 시작되었습니다.")

    try:
        while not stop_event.is_set():
            client_socket = None
            try:
                client_socket, addr = server_socket.accept()
                print(f"✅ 연결됨: {addr}")
                client_socket.settimeout(1.0)
            
                data = client_socket.recv(1024 * 1024)

                if not data:
                    print("🔌 데이터 없음")
                
                receive_data = json.loads(data.decode('utf-8'))
                
                receive_data = RequestBody(**receive_data)
                user_data = receive_data.data

                print(user_data[:5])
                print(f"📩 수신 완료: {len(user_data)} 건")

                all_data = []
                for step in user_data:
                    if stop_event.is_set():
                        break
                    
                    p_data = {
                        'timestamp': datetime.fromisoformat(step.get("ts") or step.get("timestamp")),
                        'x': step.get("x"),
                        'y': step.get("y"),
                        'deltatime': step.get("dt") or step.get("deltatime")
                    }
                    
                    result = detector.push(p_data)
                    if result:
                        m_str = result.get('macro_probability', "0%")
                        raw_e = result.get('raw_error', 0.0)
                        log_msg = f"{m_str} (err: {raw_e:.4f})"
                        if not result["is_human"]: log_msg += " 🚨"
                        
                        if log_queue: log_queue.put(log_msg)
                        else: print(log_msg)
                        all_data.append(log_msg)

                result_json = ResponseBody(
                    id = receive_data.id,
                    status = 0,
                    analysis_results = all_data
                )

                final_payload = result_json.model_dump_json().encode('utf-8')

                client_socket.sendall(final_payload)
                print(f"📤 분석 결과 {len(all_data)}건 전송 완료")

                # 버퍼 초기화
                detector.buffer = deque(maxlen=g_vars.SEQ_LEN * 2) 
            except socket.timeout:
                continue  # 🔥 정상: 아직 데이터 없음                
            except Exception as e:
                # 5. 내부 서버 에러 (status: 500)
                print(f"❌ 분석 중 에러: {e}")
                error_res = json.dumps({"status": 500, "message": str(e)}).encode('utf-8')
                client_socket.sendall(error_res)
            finally:
                if client_socket:
                    client_socket.close() 
    except Exception as e:
        print(f"❌ 서버 치명적 오류: {e}")
 
    try:
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

    server_socket.close()
    print("🛑 서버 소켓 종료")
    stop_event.set()