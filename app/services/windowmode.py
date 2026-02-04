def windowmode():
    import keyboard
    import threading
    import sys
    import ctypes
    import time

    import app.core.globals as g_vars

    from multiprocessing import Event
    from app.utilites.resource_monitoring import ResourceMonitor    
    from app.utilites.yncheck import yncheck

    user_input2 = input("chart Show? (y/n): ").lower()
    
    if not yncheck(user_input2):
        sys.exit()

    chart_Show = user_input2 == 'y'

    ctypes.windll.kernel32.SetConsoleTitleW("Inference Mode (Quit: CTRL+SHIFT+Q)")
    g_vars.init_manager()

    stop_move_event = Event()

    def trigger_stop_event():
        stop_move_event.set()
        print("\n🛑 STOP SIGNAL RECEIVED (CTRL+SHIFT+Q)")

    keyboard.add_hotkey('ctrl+shift+q', trigger_stop_event)

    def console_resource_logger(stop_ev, monitor):
        # 초기 타이틀 설정
        base_title = "Inference Mode (Quit: CTRL+SHIFT+Q)"
        
        while not stop_ev.is_set():
            stats = monitor.get_stats()
            
            # 타이틀에 들어갈 문자열 구성
            new_title = f"{base_title} | CPU: {stats['cpu']} | RAM: {stats['ram']} | GPU: {stats['gpu']}"
            
            # 실시간으로 윈도우 타이틀 변경
            ctypes.windll.kernel32.SetConsoleTitleW(new_title)
            
            time.sleep(1) # 1초 간격 갱신
        
        # 종료 시 타이틀 복구
        ctypes.windll.kernel32.SetConsoleTitleW("Inference Stopped")
        
    # 모니터 객체 생성 및 스레드 시작
    monitor = ResourceMonitor()
    res_thread = threading.Thread(
        target=console_resource_logger, 
        args=(stop_move_event, monitor), 
        daemon=True # 프로그램 종료 시 자동 종료
    )
    res_thread.start()

    user_input3 = input("🚀 Mode Select [1: 📡 Socket, 2: 📂 Load JSON, 3: 💻 Live Window]: ").strip()

    if user_input3 == "1":
        import app.services.inferece_json as inference_json
        inference_json.main(
            stop_event=stop_move_event,
            chart_Show=chart_Show,
            mode=user_input3
        ) 
    elif user_input3 == "2":
        import app.services.inferece_json as inference_json
        inference_json.main(
            stop_event=stop_move_event,
            chart_Show=chart_Show,
            mode=user_input3
        )
    elif user_input3 == "3":
        import app.services.inferece as inference
        # 인퍼런스 실행 (이 함수가 종료될 때까지 대기함)
        inference.main(
            stop_event=stop_move_event,
            chart_Show=chart_Show,
        )               
    else:
        print("❌ 잘못된 입력입니다. 1, 2, 3 중에서 선택해주세요.")
        sys.exit()