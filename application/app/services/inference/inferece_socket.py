import time
import app.core.globals as g_vars
from datetime import datetime
from multiprocessing import Queue

from app.services.inference.macro_dectector import MacroDetector
from multiprocessing import Event
from app.models.MouseDetectorSocket import ResponseBody, RequestBody
import socket
import json
from queue import Empty
from tkinter import filedialog, messagebox
import os

def main(stop_event=None, log_queue:Queue=None, chart_Show=True):
    use_existing = False
    if g_vars.init_model_path and g_vars.init_scale_path:
        if os.path.exists(g_vars.init_model_path) and os.path.exists(g_vars.init_scale_path):
            model_name = os.path.basename(g_vars.init_model_path)
            msg = f"이전에 사용한 모델을 다시 사용하시겠습니까?\n\n모델: {model_name}"
            use_existing = messagebox.askyesno("경로 재사용", msg)
        else:
            if log_queue: log_queue.put("⚠️ 이전 모델 파일이 경로에 없습니다. 새로 선택합니다.")

    # 2. '아니오'를 눌렀거나 기존 경로가 없는 경우에만 새로 선택
    if not use_existing:
        initial_dir = g_vars.scaler_path
        
        # (1) 모델 선택
        new_model_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="[1/2] 학습된 모델(.pt) 파일을 선택하세요",
            filetypes=(("PyTorch 모델", "*.pt"), ("모든 파일", "*.*"))
        )
        if not new_model_path:
            if log_queue: log_queue.put("❌ 모델 선택이 취소되었습니다.")
            return
        g_vars.init_model_path = new_model_path

        # (2) 스케일러 선택
        new_scale_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="[2/2] 해당 모델의 스케일러(.pkl) 파일을 선택하세요",
            filetypes=(("스케일러 파일", "*.pkl"), ("모든 파일", "*.*"))
        )
        if not new_scale_path:
            if log_queue: log_queue.put("❌ 스케일러 선택이 취소되었습니다.")
            return
        g_vars.init_scale_path = new_scale_path

    # 3. 최종 경로 확정 로그 (이 부분을 g_vars 사용으로 수정!)
    if log_queue:
        # local variable 대신 g_vars 값을 참조하여 에러 방지
        m_name = os.path.basename(g_vars.init_model_path)
        s_name = os.path.basename(g_vars.init_scale_path)
        log_queue.put(f"📂 로드 완료:\n- 모델: {m_name}\n- 스케일러: {s_name}")

    if stop_event is None:
        stop_event = Event()

    # Detector 초기화
    detector = MacroDetector(
        model_path=g_vars.init_model_path,
        seq_len=g_vars.SEQ_LEN,
        threshold=g_vars.threshold,
        chart_Show=chart_Show,
        stop_event=stop_event,
        scale_path=g_vars.init_scale_path
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
                    
                    detector.push(p_data)

                # 추론 시작
                send_data = detector._infer()                    

                result_json = ResponseBody(
                    id = receive_data.id,
                    status = 0,
                    analysis_results = send_data
                )

                final_payload = result_json.model_dump_json().encode('utf-8')

                client_socket.sendall(final_payload)
                
                # 버퍼 초기화
                detector.buffer.clear()
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
        detector.buffer.clear()
        if log_queue:
            log_queue.put("🛑 Detector 종료")
        else:
            print("🛑 Detector 종료")
        try:
            while True:
                g_vars.CHART_DATA.get_nowait()
        except Empty:
            pass            
        stop_event.set()

    if log_queue:
        log_queue.put("🛑 Macro Detector Stopped")
    else:
        print("🛑 Macro Detector Stopped")

    server_socket.close()
    print("🛑 서버 소켓 종료")
    stop_event.set()