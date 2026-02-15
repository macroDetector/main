import app.core.globals as g_vars\

try:
    import app.core.globals as g_vars
    from app.gui.handlers import UIHandler
except ImportError:
    from queue import Queue
    import os, sys

    class Mock: pass
    globals = Mock()
    g_vars.SEQ_LEN, g_vars.STRIDE, g_vars.d_model = 100, 10, 128
    g_vars.num_layers, g_vars.lr, g_vars.tolerance, g_vars.threshold = 2, 0.0, 0.0001, 0.5
    g_vars.LOG_QUEUE = Queue()

    class UIHandler:
        def __init__(self, ev, parent=None): # parent 인자 추가
            self.ev = ev
            self.parent = parent
            self.tray = None # 트레이 객체 저장용
            
        def start_record(self, **kwargs): 
            g_vars.LOG_QUEUE.put("🎥 Recording Started (Mock Mode)")
            
        def start_train(self): 
            g_vars.LOG_QUEUE.put("🧠 Training Started (Mock Mode)")
            
        def start_inference(self): 
            g_vars.LOG_QUEUE.put("⚡ Inference Started (Mock Mode)")
            
        def make_plot(self, user=False): 
            g_vars.LOG_QUEUE.put(f"📊 Plotting {'User' if user else 'Bot'} path... (Mock Mode)")

        def setup_tray(self):
            """Mock 환경에서는 실제 UI를 띄우지 않고 로그만 남깁니다."""
            g_vars.LOG_QUEUE.put("시스템 트레이 설정 완료 (Mock Mode)")

        def exit_application(self):
            """애플리케이션 종료 로그를 남기고 프로세스 종료"""
            g_vars.LOG_QUEUE.put("프로그램 종료 중... (Mock Mode)")
            self.ev.set()
            import os
            os._exit(0)

        def update_parameters(self, data_dict):
            """Mock 환경에서도 UI로부터 전달받은 설정값을 globals에 적용하고 .env에 저장"""
            try:
                # 1. 전역 변수 업데이트 (타입 캐스팅 포함)
                g_vars.SEQ_LEN = int(data_dict.get('SEQ_LEN', 100))
                g_vars.STRIDE = int(data_dict.get('STRIDE', 10))
                g_vars.d_model = int(data_dict.get('D_MODEL', 128))
                g_vars.num_layers = int(data_dict.get('LAYERS', 2))
                g_vars.lr = float(data_dict.get('LR', 0.0))
                g_vars.threshold = float(data_dict.get('THRES', 0.5))
                g_vars.tolerance = float(data_dict.get('TOLE', 0.0001))

                # 2. .env 내용 생성
                env_content = (
                    f"SEQ_LEN={g_vars.SEQ_LEN}\n"
                    f"STRIDE={g_vars.STRIDE}\n"
                    f"D_MODEL={g_vars.d_model}\n"
                    f"NUM_LAYERS={g_vars.num_layers}\n"
                    f"LEARNING_RATE={g_vars.lr}\n"
                    f"THRESHOLD={g_vars.threshold}\n"
                    f"TOLERANCE={g_vars.tolerance}\n"
                )

                # 3. 파일 쓰기 (실행 파일 경로 고려)
                base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
                with open(os.path.join(base_path, ".env"), "w", encoding="utf-8") as f:
                    f.write(env_content)

                g_vars.LOG_QUEUE.put("✅ [HANDLER] Parameters successfully synchronized to .env")
                return True

            except Exception as e:
                if hasattr(globals, 'LOG_QUEUE'):
                    g_vars.LOG_QUEUE.put(f"❌ [HANDLER ERROR] Failed to update: {e}")
                return False
