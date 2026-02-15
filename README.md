# 🤖 Model Architecture: Transformer Macro Autoencoder

Transformer 기반의 오토인코더(Autoencoder) 구조
- 정상 패턴: 모델이 높은 정확도로 복원하여 재구성 오차 0에 수렴합니다.
- 이상 패턴(매크로): 모델이 학습하지 못한 패턴이므로 복원 능력이 떨어져 재구성 오차가 높게 발생합ㄴ다.

Detection Logic
- Normal Patterns: The model reconstructs these with high precision, causing the reconstruction error to converge to zero.
- Anomalous Patterns (Macro): Since these are patterns the model has not encountered during training, the reconstruction capability decreases, resulting in a high reconstruction error.

![Architecture Diagram](./public/Architecture.png)

# 정식 1.0.0 버전 출시 전까지 기능 개선 및 안정화를 위해 빈번한 업데이트가 진행될 예정입니다.
# Frequent updates are expected for feature enhancement and stabilization until the official v1.0.0 release.

# 🚀 Macro Detector Update (Ver 0.0.5)

## 📊 주요 업데이트 사항 (Major Updates)

### 1. 데이터 구조 및 저장 방식 개선
* **JSON 통합 파이프라인:** 기존 PostgreSQL 지원을 삭제하고 모든 데이터를 **JSON 포맷**으로 통합하여 데이터 처리 속도를 높이고 이식성을 극대화했습니다.
* **경량화:** 무거운 데이터베이스 의존성을 제거하여 로컬 환경에서도 빠른 분석과 학습이 가능합니다.

### 2. 가우시안 지표 엔지니어링 (Feature Engineering)
단순 좌표 추적을 넘어, 움직임의 **'DNA(통계적 형상)'**를 분석하도록 지표 체계를 완전히 재편했습니다.
* **가우시안 핵심 (Gaussian Core):**
  - **평균 (Mean, $\mu$):** 물리량의 중심 이동 경로 추적.
  - **표준편차 (Std, $\sigma$):** 움직임의 변동성 및 불안정성(Spread) 감시.
* **형태 및 무질서도 분석:**
  - **왜도 (Skewness):** 데이터의 비대칭성 및 특정 방향으로의 쏠림(이상치) 탐지.
  - **거칠기 (Roughness):** 미세 떨림 및 기계적인 불규칙성 포착.
* **직선도 분석 (Straightness Suite):** 직선도에 대해 평균, 표준편차, 왜도, 거칠기를 모두 적용하여 매크로 특유의 선형 운동 탐지 기능을 강화했습니다.

### 3. 학습 모델 및 로직 고도화
* **정규 분포 학습:** 개별 데이터 포인트가 아닌, **Chunk_size(100)** 단위의 가우시안 분포 특성을 학습하도록 변경되었습니다.
* **손실 함수 최적화:** **MAE(Mean Absolute Error)**를 도입하여 이상치(Outlier)에 대한 모델의 저항력을 높였습니다.
* **도메인 특화 모델 (Domain Specialization):** - **FPS 모드:** 급격한 에임 및 반동 패턴 최적화.
  - **웹/UI 모드:** 자동 클릭 및 단순 반복 경로 최적화.

### 4. 추론 및 시각적 모니터링
* **가우시안 실시간 차트:** 텍스트 기반 로그 탐지를 삭제하고, **가우시안 분포 곡선**을 통한 시각적 모니터링을 제공합니다.
* **이상 징후 가시화:** 이상 행동 발생 시 종 모양의 그래프가 급격히 벌어지거나 이동하는 모습을 직관적으로 확인 가능합니다.


---

### 📦 Libray 지원

```bash
pip install git+https://github.com/qqqqaqaqaqq/mouseMacroLibrary.git

---