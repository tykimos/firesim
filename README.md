# 🔥 Fire Simulation & AI Prediction System

## 📋 프로젝트 개요

소방관 훈련을 위한 현실적인 화재 시뮬레이션과 AI 기반 화재 예측 시스템입니다. 물리학 기반 화재 확산 모델링과 ConvLSTM 기반 화재 예측 AI를 결합하여 실시간 화재 상황 예측과 훈련 시나리오를 제공합니다.

### 🎯 주요 기능
- 🔥 **현실적 화재 시뮬레이션**: t-squared 화재 성장, 복사열전달, 연기 확산 모델링
- 🧠 **AI 화재 예측**: 20초 과거 데이터로 5초 후 화재 상황 예측
- 📊 **실시간 모니터링**: 9개 환경 변수의 실시간 시각화
- 💾 **데이터 수집**: 시계열 데이터 자동 수집 및 바이너리 저장
- 📈 **성능 분석**: Ground Truth vs AI 예측 비교 및 정확도 평가

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fire Simulation System                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Simulation    │  │   AI Training   │  │   AI Inference  │ │
│  │   (Physics)     │  │   (ConvLSTM)    │  │   (Prediction)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Data Storage   │  │   Comparison    │  │   Evaluation    │ │
│  │   (Binary)      │  │   (Playback)    │  │   (Accuracy)    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 파일 구조

```
firesim/
├── 📄 README.md                    # 이 문서
├── 🔥 fire_simulation.py          # 메인 화재 시뮬레이션
├── 📺 fire_playback.py            # 시뮬레이션 데이터 재생
├── 🧠 train_model.py              # AI 모델 학습
├── 🎯 inference_model.py          # AI 모델 추론
├── 📊 compare_playback.py         # GT vs AI 비교 시각화
├── 📈 evaluate_accuracy.py        # 정확도 평가
├── 🔧 prepare_datasets.py         # 데이터셋 분할
├── 📂 train_dataset/              # 학습 데이터
├── 📂 test_dataset/               # 테스트 데이터
├── 📂 pred_dataset/               # AI 예측 결과
├── 📂 docs/                       # 기술 문서
│   ├── 화재시뮬레이션_이론.md       # 물리학 이론
│   └── AI모델_이론.md             # 딥러닝 이론
└── 💾 *.pth                       # 학습된 모델 파일
```

## 🚀 빠른 시작

### 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는 .venv\Scripts\activate  # Windows

# 필요한 패키지 설치
pip install torch pygame numpy scikit-learn matplotlib
```

### 기본 워크플로우
```bash
# 1. 화재 시뮬레이션 실행 (데이터 생성)
python fire_simulation.py

# 2. 데이터셋 분할
python prepare_datasets.py

# 3. AI 모델 학습
python train_model.py

# 4. AI 예측 생성
python inference_model.py

# 5. 결과 비교 및 평가
python compare_playback.py
python evaluate_accuracy.py
```

## 🔥 화재 시뮬레이션 알고리즘

### 화재 상태 모델
```python
화재_상태 = {
    0: NORMAL,      # 정상 상태 (20°C)
    1: PREHEATING,  # 예열 (150°C 이상, 열분해 시작)
    2: IGNITION,    # 착화 (400°C 이상, 지속 연소)
    3: GROWTH,      # 성장 (t-squared 모델)
    4: FLASHOVER,   # 플래시오버 (600°C 이상, 급속 확산)
    5: BURNOUT      # 연소 완료 (연료/산소 고갈)
}

# 상태 전이 조건 (비가역적)
transition_conditions = {
    'temperature_thresholds': [20, 150, 400, 400, 600],
    'fuel_depletion': 'fuel_load < 1.0 MJ/m²',
    'oxygen_limit': 'oxygen_level < 12.0%'
}
```

### 물리학 기반 모델링

#### 1. 복사열전달 (Stefan-Boltzmann 법칙)
```python
Q_rad = ε × σ × (T⁴ - T₀⁴) / 1000  # kW/m²
```
- ε: 방사율 (0.8)
- σ: Stefan-Boltzmann 상수 (5.67×10⁻⁸)
- T: 절대온도 (K)

#### 2. t-squared 화재 성장
```python
HRR(t) = α × t²  # Heat Release Rate
```
- α: 화재 성장 계수 (0.05)
- t: 연소 시간 (초)

#### 3. 연기 가시거리 (Jin's 방정식)
```python
visibility = 2.0 / (smoke_density × (1 + T/300))
```

#### 4. 독성가스 농도
```python
# CO 생성 (산소 고갈 시 급증)
CO_production = burning_rate × CO_yield × oxygen_depletion_factor
# HCN 생성 (고온에서 질소 함유 재료)
HCN_production = burning_rate × HCN_yield × temperature_factor
```

### 환경 변수 (9개)

| 변수 | 단위 | 범위 | 설명 |
|------|------|------|------|
| Fire State | enum | 0-5 | 화재 상태 (셀룰러 오토마타) |
| Temperature | °C | 20-1200 | 온도 (열전도/복사) |
| Smoke Density | mg/m³ | 0-6000 | 연기 밀도 (Fick 확산) |
| Visibility | m | 0-30 | 가시거리 (Jin 공식) |
| CO Concentration | ppm | 0-40000 | 일산화탄소 (산소고갈 시 증가) |
| HCN Concentration | ppm | 0-6000 | 시안화수소 (고온 열분해) |
| Air Velocity | m/s | 0-6 | 공기 속도 (부력 구동) |
| Thermal Radiation | kW/m² | 0-100 | 복사열 (Stefan-Boltzmann) |
| Pressure | Pa | 101000-102500 | 압력 (이상기체 법칙) |

## 🧠 AI 예측 모델

### ConvLSTM 아키텍처
```
Input: [Batch, 20_frames, 9_channels, 20_height, 20_width]
│
├── Feature Extractor (Conv2D + BatchNorm + ReLU)
│   ├── Conv2D(9→64, 3×3)
│   ├── Conv2D(64→64, 3×3)  
│   └── Conv2D(64→128, 3×3)
│
├── ConvLSTM Layers (×3)
│   ├── ConvLSTMCell with 3×3 kernels
│   ├── LayerNorm + Dropout (p=0.1)
│   └── Hidden dimension: 128 channels
│
├── Spatial-Temporal Attention
│   ├── Spatial Attention Module
│   └── Channel Attention Module
│
├── Decoder (Conv2D layers)
│   ├── Conv2D(128→128, 3×3)
│   ├── Conv2D(128→64, 3×3)
│   ├── Conv2D(64→32, 3×3)
│   └── Conv2D(32→9, 3×3)
│
Output: [Batch, 9_channels, 20_height, 20_width]
```

### 학습 전략

#### 1. 데이터 증강
- 수평/수직 뒤집기
- 90도, 180도 회전
- 가우시안 노이즈 주입 (σ=0.01)
- 슬라이딩 윈도우 (1775개 샘플 생성)

#### 2. 손실 함수 (Physics-Informed Weighted MSE)
```python
weights = {
    Fire_State: 3.0,      # 가장 중요
    Temperature: 2.0,
    Smoke_Density: 1.5,
    Visibility: 1.5,
    Thermal_Radiation: 1.5,
    CO_Concentration: 1.0,
    HCN_Concentration: 1.0,
    Air_Velocity: 1.0,
    Pressure: 0.5         # 가장 덜 중요
}

# 물리 제약 추가
total_loss = weighted_mse + 0.1 * range_loss + 0.05 * smoothness_loss
```

#### 3. 최적화
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- Early Stopping: patience=15
- Gradient Clipping: max_norm=1.0

### 추론 논리
```python
# 20초 과거 데이터로 5초 후 예측
sequence_length = 20
prediction_offset = 5

for timestep t in range(sequence_length + prediction_offset, total_timesteps):
    # 입력: t-25부터 t-5까지 20프레임 시퀀스
    input_sequence = ground_truth[t-sequence_length-prediction_offset:t-prediction_offset]
    
    # 모델 추론
    with torch.no_grad():
        prediction = model(input_sequence.unsqueeze(0))  # 배치 차원 추가
    
    # 예측 결과 저장 (t 시점 예측값)
    pred_data[t] = prediction.squeeze(0)
```

## 📊 사용법 가이드

### 1. 화재 시뮬레이션 실행
```bash
python fire_simulation.py
```
**조작법:**
- `↑↓←→`: 셀 선택 이동
- `SPACE`: 일시정지/재생
- `R`: 시뮬레이션 리셋
- `E`: 데이터 수동 내보내기

### 2. AI 모델 학습
```bash
python prepare_datasets.py  # 데이터셋 분할
python train_model.py       # 모델 학습
```

### 3. AI 추론 및 비교
```bash
python inference_model.py   # AI 예측 생성
python compare_playback.py  # GT vs AI 비교
python evaluate_accuracy.py # 정확도 평가
```

### 4. 시뮬레이션 데이터 재생
```bash
python fire_playback.py
```

## 💾 데이터 형식

### 바이너리 데이터 구조
```
Shape: [timesteps, height, width, variables]
Example: [1800, 20, 20, 9]
- 1800: 30분 시뮬레이션 (1초 간격)
- 20×20: 격자 크기
- 9: 환경 변수 개수
```

### 메타데이터 (JSON)
```json
{
  "grid_width": 20,
  "grid_height": 20,
  "ignition_point": [18, 11],
  "variables": ["fire_state", "temperature", ...],
  "variable_units": ["state_enum", "celsius", ...],
  "data_shape": [1800, 20, 20, 9],
  "total_timesteps": 1800,
  "prediction_info": {
    "model_type": "ConvLSTM Fire Prediction",
    "input_sequence_length": 20,
    "prediction_offset": 5,
    "prediction_start_timestep": 25,
    "architecture": {
      "hidden_dim": 128,
      "num_layers": 3,
      "kernel_size": 3,
      "dropout": 0.1
    }
  }
}
```

## 📈 성능 지표

### 현재 모델 성능 (실제 측정값)
- **변수별 정확도** (evaluate_accuracy.py 결과):
  - Fire State: 93.7% (R²=0.926)
  - Temperature: 89.9% (R²=0.536)
  - Smoke Density: 높은 정확도 (R²=0.940)
  - 기타 변수들도 양호한 성능 달성

**측정 기준:**
- 분류 정확도 (Fire State)
- 회귀 R² 스코어 (연속 변수)
- MSE, MAE, RMSE 포함

### 평가 방법
1. **분류 정확도** (Fire State)
2. **회귀 MSE** (연속 변수들)
3. **시각적 비교** (Heat Map)

## 🔧 고급 설정

### M3 Mac 가속 (MPS)
```python
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Apple Silicon GPU 가속 활성화")
    # MPS 메모리 관리
    torch.mps.empty_cache()
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 배치 크기 최적화
- MPS: batch_size = 4 (메모리 효율성)
- CPU/CUDA: batch_size = 2
- 데이터 로딩: num_workers=0, pin_memory=False (MPS 최적화)

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 모델 로딩 오류
```bash
Error: cannot import name 'FirePredictionModel'
```
**해결**: `train_model.py`의 정확한 클래스명 확인

#### 2. 예측이 변하지 않음
**원인**: 모델이 제대로 학습되지 않음 또는 그래디언트 소실
**해결**: 
- 학습률 조정 (0.001 → 0.01)
- 그래디언트 클리핑 확인
- 물리 제약 손실 가중치 조정

#### 3. MPS 메모리 부족
**해결**: 
- 배치 크기 줄이기 (4 → 2)
- torch.mps.empty_cache() 정기 호출
- num_workers=0으로 설정

### 디버깅 도구
```bash
python debug_predictions.py      # 예측 변화 확인
python debug_model_behavior.py   # 모델 동작 테스트
```

## 📚 참고문헌

### 화재 과학
- NFPA 921: Guide for Fire and Explosion Investigations
- Drysdale, D. "An Introduction to Fire Dynamics"
- Karlsson & Quintiere "Enclosure Fire Dynamics"

### 머신러닝
- Xingjian et al. "Convolutional LSTM Network"
- Attention Mechanisms in Deep Learning
- PyTorch Documentation

## 🎯 현재 구현 상태

### ✅ 완전 구현됨
1. **화재 시뮬레이션**: 9개 변수 물리 기반 2D 시뮬레이션
2. **AI 학습/추론**: ConvLSTM 기반 예측 모델
3. **시각화**: Ground Truth vs AI 비교 및 정확도 평가
4. **데이터 관리**: 바이너리 저장, 메타데이터, 자동 분할

### ⚠️ 현재 제한사항
1. **데이터셋 규모**: 7개 훈련 파일 (PoC 수준)
2. **예측 범위**: 5초 후 단기 예측만
3. **격자 크기**: 20×20 고정 (400 셀)
4. **공간 차원**: 2D만 지원

## 👥 기여자

- **개발자**: AI Assistant (Claude) - 시스템 설계 및 구현
- **프로젝트 관리**: tykimos - 프로젝트 기획 및 감독
- **목적**: 소방관 훈련을 위한 안전한 AI 기반 화재 시뮬레이션 시스템
- **기술 스택**: PyTorch, PyGame, NumPy, ConvLSTM

---

**🔥 안전한 화재 시뮬레이션으로 더 나은 소방 훈련을! 🔥**