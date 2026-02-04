# 🤖 Model Architecture: Transformer Macro Autoencoder

Transformer 기반의 오토인코더(Autoencoder) 구조
- 정상 패턴: 모델이 높은 정확도로 복원하여 재구성 오차 0에 수렴합니다.
- 이상 패턴(매크로): 모델이 학습하지 못한 패턴이므로 복원 능력이 떨어져 재구성 오차가 높게 발생합ㄴ다.

- Feature Embedding : 5차원의 입력 피처(x, y, dist 등)를 d_model(64차원)의 고차원 벡터로 확장하여 복잡한 상관관계를 학습할 준비를 합니다.
- Positional Encoding : Transformer는 RNN과 달리 순서 정보가 없으므로, 시퀀스 내 각 위치 정보($1^{st}, 2^{nd}, ...$)를 나타내는 벡터를 더해줍니다.
- Transformer Encoder : Multi-Head Self-Attention 메커니즘을 통해 시퀀스 전체를 동시에 훑으며, 과거의 움직임이 현재에 미치는 영향을 파악합니다.
- Linear Decoder : 인코더가 뽑아낸 추상적인 특징들을 다시 원래의 5개 피처 차원으로 복원합니다.

Detection Logic
- Normal Patterns: The model reconstructs these with high precision, causing the reconstruction error to converge to zero.
- Anomalous Patterns (Macro): Since these are patterns the model has not encountered during training, the reconstruction capability decreases, resulting in a high reconstruction error.

- Feature Embedding: Expands the 5-dimensional input features (e.g., $x, y, dist$) into a high-dimensional vector of $d_{model}$ (64 dimensions) to prepare the model for learning complex correlations.
- Positional Encoding: Since Transformers do not inherently process sequential order like RNNs, this adds vectors that represent the positional information ($1^{st}, 2^{nd}, \dots$) within the sequence.
- Transformer Encoder: Utilizes the Multi-Head Self-Attention mechanism to scan the entire sequence simultaneously, capturing how past movements influence the present state.
- Linear Decoder: Reconstructs the abstract features extracted by the encoder back into the original 5-feature dimensions.

![Architecture Diagram](./public/Architecture.png)
---
# 🚀 Macro Detector Update Ver 0.0.3

## 🛠 추가 및 변경 기능
* **소켓 모드(Socket Mode) 도입**
    * `Inference Mode`에서만 활성화됩니다.
    * 서버 주소: `localhost:52341` (TCP/IP)
* **JSON 모드 UI 편의성 개선**
    * `PLOT USER PATH`: 클릭 시 즉시 파일 탐색기(File Dialog) 실행.
    * `Json Data Inference`: 실행 전 추론할 파일을 직접 선택하도록 변경.
* **데이터 저장 로직 변경**
    * 기존: `append` (기존 파일에 추가)
    * 변경: **날짜/시간별 신규 파일 생성** (데이터 무결성 및 관리 편의성 증대)

## 🛠 New Features & Enhancements
* **Added Socket Mode**
    * Exclusively enabled in `Inference Mode` for real-time data processing.
    * Configuration: `server_socket.bind(("localhost", 52341))`
* **Improved JSON Mode UI Workflow**
    * `PLOT USER PATH` now triggers a native File Dialog for easier navigation.
    * `Json Data Inference` updated to prompt for a file selection before execution.
* **Storage Logic Overhaul**
    * Switched from `append` to **Timestamp-based unique file generation** for mouse data.

---
# 🚀 Macro Detector Update Ver 0.0.2

## 🛠 주요 기능 및 개선 사항 (Key Features & Enhancements)

### 1. 물리 기반 고정밀 트래킹 (High-Fidelity Physics-Based Tracking)
기존의 고정 시간 샘플링 방식에서 벗어나, 하드웨어의 실제 움직임을 포착하는 **Event-driven** 모델로 마이그레이션되었습니다.
* **OS 레벨 이벤트 리스너:** `pynput.mouse.Listener`를 통해 하드웨어 인터럽트를 직접 수신합니다.
    * **Old:** 0.02s (feat tolerance) 간격의 강제 샘플링 (디지털적으로 정형화된 데이터)
    * **New:** 마우스가 움직일 때마다 발생하는 실제 $\Delta t$ (예: 0.0209s)를 기록하여 **"Physical Truth"**를 확보합니다.
* **영향:** 인간 특유의 미세한 가속도 곡선, 유기적인 타이밍 변화, 그리고 물리적 지터(Jitter)를 보존하여 AI의 판별력을 극대화합니다. 
* **웹페이지:** 웹페이지 특유의 16.66ms 를 고려하여 설계되었습니다.
* **Tolerance** UI 내 최소 간격을 조정할 수 있습니다.

### 2. 피처 엔지니어링 고도화 (Feature Engineering)
단순 좌표 분석을 넘어, 물리 법칙을 적용한 다차원 변수를 추출합니다.
* **운동 파생 변수:** 고정밀 타임스탬프를 기반으로 다음 지표를 산출합니다.
    * **Velocity (속도)**, **Acceleration (가속도)**, **Jerk (가속도 변화율)** 등
* **엔트로피 분석:** 매크로의 선형적 움직임과 대비되는 인간의 '유기적 불규칙성'을 수치화하여 피처 공간(Feature Space)을 확장했습니다.

### 3. 자동화된 모델 최적화 (Automated Optimization)
환경에 구애받지 않는 범용 탐지를 위해 판단 로직을 자동화했습니다.
* **Auto-Threshold:** 학습 데이터의 분포를 분석하여 최적의 탐지 임계값을 자동으로 설정합니다. 추론시 임계값의 1.05배 기준으로 계산됩니다.
* **Post-Analysis Mode:** 실시간 탐지 외에도 기존에 저장된 JSON 로그를 분석하는 포렌식 기능을 지원합니다.
* **중앙 관리 구조:** `config.json`을 통해 모든 하이퍼파라미터를 통합 관리합니다.

### 4. 시스템 안정성 강화 (Resilience & Stability) 🛡️
* **Asynchronous Queue:** 데이터 수집과 추론 로직을 분리하여 CPU 부하 상황에서도 마우스 끊김(Stuttering)이 발생하지 않습니다.
* **Protection Mode:** 시스템 권한 창(작업 관리자 등) 접근 시에도 충돌 없이 기록을 유지하는 Fail-safe 프로토콜을 적용했습니다.

## ✨ UI/UX 개선
* **Dark-themed UI:** 시각적 가독성을 높인 현대적 디자인.
* **Tray Integration:** 백그라운드 구동을 위한 시스템 트레이 아이콘 지원.
* **Real-time Feedback:** 감지 시 사이렌 아이콘과 함께 실시간 확률(%) 출력.

## 🛠 Key Features & Enhancements

### 1. High-Fidelity Physics-Based Tracking
Migrated from fixed-interval polling to an **OS-level Event-driven** model to capture the "Physical Truth" of hardware input.
* **OS-Level Event Listener:** Utilizes `pynput.mouse.Listener` to receive direct hardware interrupts.
    * **Old:** Forced sampling at 0.02s intervals (Digitized/Synthetic data).
    * **New:** Records high-precision $\Delta t$ (e.g., 0.0209s) for every hardware event.
* **Web-Optimized Design:** Specifically engineered to account for the $16.66ms$ refresh cycles (60Hz) typical of web environments.
* **Configurable Tolerance:** Added a UI-based setting to adjust the minimum temporal interval (Tolerance) for stable inference.
* **Impact:** Preserves human-centric micro-timing, organic acceleration curves, and physical jitter—critical factors for AI-based differentiation.

### 2. Advanced Feature Engineering
Extracts multi-dimensional variables by applying laws of physics to raw coordinate data.
* **Motion Derivatives:** Calculates high-precision metrics based on precise timestamps:
    * **Velocity ($v$)**, **Acceleration ($a$)**, **Jerk ($j$ - rate of change of acceleration)**.
* **Entropy & Jitter Analysis:** Quantifies "organic irregularity" vs. the "linear rigidity" of macros to expand the feature space.

### 3. Automated Model Optimization
Standardized detection logic to ensure universal performance across different hardware environments.
* **Auto-Threshold Calculation:** Automatically determines the optimal detection threshold based on training data distribution. During inference, the system operates on a $1.05\times$ threshold margin.
* **Post-Analysis Mode:** Supports JSON data inference for forensic analysis of pre-recorded logs.
* **Centralized Configuration:** All hyperparameters and derived thresholds are managed via a single `config.json` file.

### 4. System Resilience & Stability 🛡️
* **Asynchronous Queue Architecture:** Decouples the Listener (Capture) from the Main Loop (Inference), eliminating mouse stuttering even under high CPU load.
* **Protection Mode (Fail-Safe):** Integrated protocols to maintain stable recording and prevent crashes when interacting with restricted system windows (e.g., Task Manager).

## ✨ UI/UX Improvements
* **Modern Dark Theme:** Refined dashboard with a focus on visual clarity and reduced eye strain.
* **System Tray Integration:** Added "Minimize to Tray" support for seamless background monitoring.
* **Real-time Detection Feedback:** Instant visual alerts using siren emojis and real-time probability percentages (%).

## 📊 Quick Comparison

| Feature | Polling System (Old) | Event Listener (New) |
| :--- | :--- | :--- |
| **Trigger** | Clock Timer (Fixed) | Hardware Interrupt (Physical) |
| **Time Delta ($\Delta t$)** | Normalized (Forced 0.02s) | Raw High-Precision (Actual Physics) |
| **Data Quality** | Lossy / Synthetic | High-Fidelity / Organic |
| **Human Jitter** | Smoothed Out (Filtered) | **Captured Accurately (Essential for AI)** |

---

## 🚀 Update Ver 0.0.1

### 🔧 Features
* **CLI Mode Expansion:** Inference Mode now officially supports both **Windows CMD** environments for broader compatibility.
* **Portable Release:** Executables are now bundled and distributed as **ZIP archives** via PyInstaller, allowing for easy deployment without complex installation.

### ⌨️ Shortcuts & Commands
* **Inference Mode (CLI):** - `Start` => `Inference Mode` => `Yes`
* **Inference Mode (UI):** - `UI` => `Inference Mode` => `No`
* **Emergency Quit:** `Ctrl + Shift + Q`

![Cmdupdate](./public/Cmdupdate.png)

---

## 📂 Data Management
* **Database Support:** Efficient data handling using **PostgreSQL** and **JSON** formats.

## 🛠 Installation
* To install the required dependencies, run the following command:
  ```bash
  pip install -r requirements.txt

## 사용 설명서 (Manual)
Manual.pptx

## 영상
[![실행 영상](https://img.youtube.com/vi/iwi31PxQc3I/0.jpg)](https://youtu.be/iwi31PxQc3I)