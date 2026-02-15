def update_parameters(data_dict:dict):
        try:
            import os, sys, json
            import app.core.globals as g_vars 

            base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
            config_dir = os.path.join(base_path, "config")
            config_path = os.path.join(config_dir, "config.json")

            if not os.path.exists(config_dir):
                os.makedirs(config_dir)

            # 1. 기존 설정 불러오기
            current_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    current_config = json.load(f)


            # 2. UI 파라미터 업데이트 (기존 로직)
            mapping = {
                'SEQ_LEN': ('SEQ_LEN', int),
                'STRIDE': ('STRIDE', int),
                'D_MODEL': ('d_model', int),
                'LAYERS': ('num_layers', int),
                'LR': ('lr', float),
                'THRES': ('threshold', float),
                'TOLE': ('tolerance', float),
                'N_HEAD' : ('n_head', int),
                'BATCH' : ('batch_size', int),
                "EPOCH" : ('epoch', int),
                "PATIENCE" : ('patience', int),
                "WEIGHT" : ('weight_decay', float),
                "FEED" : ('dim_feedforward', int),
                "DROP" : ('dropout', float),
                "WEIGHT_THRES" : ('weight_threshold', float),
                "IMPROVEMENT_CUT" : ('improvement_val_loss_cut', float),
                "CHUNK_SIZE" : ('chunk_size', int)
            }

            for ui_key, (json_key, dtype) in mapping.items():
                val = data_dict.get(ui_key)
                if val is not None:
                    try:
                        c_val = dtype(val)
                        current_config[json_key] = c_val
                        setattr(g_vars, json_key, c_val) 
                    except: continue

            # 3. [추가] CLIP_BOUNDS 데이터가 있다면 직접 업데이트
            # 학습 코드에서 update_parameters({"CLIP_BOUNDS": {...}}) 식으로 호출 가능
            if "CLIP_BOUNDS" in data_dict:
                c_bounds = data_dict["CLIP_BOUNDS"]
                current_config["CLIP_BOUNDS"] = c_bounds
                setattr(g_vars, "CLIP_BOUNDS", c_bounds) # 메모리 반영

            # 4. JSON 파일 저장
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(current_config, f, indent=4, ensure_ascii=False)

            g_vars.LOG_QUEUE.put("✅ [SYSTEM] Configuration & Bounds saved")

            with g_vars.lock:
                g_vars.GLOBAL_CHANGE = False

            return True

        except Exception as e:
            if hasattr(g_vars, 'LOG_QUEUE'):
                g_vars.LOG_QUEUE.put(f"❌ [STORAGE ERROR] {e}")
            return False