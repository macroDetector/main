import psutil
import os
import torch

class ResourceMonitor:
    def __init__(self):
        # 현재 프로세스 객체
        self.process = psutil.Process(os.getpid())
        # CPU 초기값 방지
        self.process.cpu_percent(interval=None)
        
        # GPU 사용 가능 여부 미리 체크
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            # 장치 이름 확인용 (디버깅)
            self.gpu_name = torch.cuda.get_device_name(0)

    def get_stats(self):
        try:
            # 1. CPU: 프로세스 점유율 / 코어 수
            raw_cpu = self.process.cpu_percent(interval=None)
            cpu_usage = raw_cpu / psutil.cpu_count()
            
            # 2. RAM: 메인 메모리 사용량 (MB)
            ram_mb = self.process.memory_info().rss / (1024 * 1024)

            # 3. GPU: PyTorch 전용 함수 사용
            gpu_display = "N/A"
            if self.gpu_available:
                try:
                    # 현재 이 프로세스가 할당(Allocated)해서 사용 중인 메모리
                    used_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
                    # 캐시/예약(Reserved)된 전체 메모리 (실제 점유 중인 크기)
                    reserved_mem = torch.cuda.memory_reserved(0) / (1024 * 1024)
                    
                    # '사용량 / 예약량' 형태로 표시
                    gpu_display = f"{used_mem:.1f} / {reserved_mem:.0f} MB"
                except:
                    gpu_display = "GPU Error"
            
            return {
                "cpu": f"{cpu_usage:.1f}%",
                "ram": f"{ram_mb:.1f} MB",
                "gpu": gpu_display
            }
        except Exception:
            return {"cpu": "0.0%", "ram": "0.0 MB", "gpu": "N/A"}