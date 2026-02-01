# 🤖 Model Architecture: Transformer Macro Autoencoder

Transformer 기반의 오토인코더(Autoencoder) 구조
- 정상 패턴: 모델이 높은 정확도로 복원하여 재구성 오차 0에 수렴합니다.
- 이상 패턴(매크로): 모델이 학습하지 못한 패턴이므로 복원 능력이 떨어져 재구성 오차가 높게 발생합ㄴ다.

- Feature Embedding : 5차원의 입력 피처(x, y, dist 등)를 d_model(64차원)의 고차원 벡터로 확장하여 복잡한 상관관계를 학습할 준비를 합니다.
- Positional Encoding : Transformer는 RNN과 달리 순서 정보가 없으므로, 시퀀스 내 각 위치 정보($1^{st}, 2^{nd}, ...$)를 나타내는 벡터를 더해줍니다.
- Transformer Encoder : Multi-Head Self-Attention 메커니즘을 통해 시퀀스 전체를 동시에 훑으며, 과거의 움직임이 현재에 미치는 영향을 파악합니다.
- Linear Decoder : 인코더가 뽑아낸 추상적인 특징들을 다시 원래의 5개 피처 차원으로 복원합니다.

---
# 지원 프로그램
- postgres
- json

# 필수 파일
.env
```
# 기록기
# postgres => postgres, json => json
Recorder=json

# posgres를 사용 시 기입
DB_HOST=-
DB_USER=-
DB_PASSWORD=-
DB_NAME=-
DB_PORT=-

# 필수 입력
SEQ_LEN=100
STRIDE=50
JsonPath=./
threshold=0.7
d_model=256
num_layers=3
dropout=0.3
batch_size=64
lr=0.0005
```

# 설치 목록
```
pynput
torch
psycopg2-binary
SQLAlchemy
pydantic_settings
pyautogui
matplotlib
numpy
pyqtgraph 
PySide6
PyQt6
keyboard
```

명령어
```
pip install -r requirements.txt
```

# 사용 설명서
사용설명서.txt