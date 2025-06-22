# Semantic Vector-Quantized Variational Autoencoder (SVQ)

이 프로젝트는 "Structured World Modeling via Semantic Vector Quantization" 논문의 핵심 아이디어를 PyTorch와 Hydra 프레임워크를 사용하여 구현한 것입니다.

## 개요

SVQ(Semantic Vector-Quantized Variational Autoencoder)는 객체 중심 학습과 벡터 양자화를 결합하여 시맨틱 신경망 이산 표현을 학습하는 모델입니다. 기존의 VQ-VAE와 달리, SVQ는 패치 수준이 아닌 객체 수준에서 의미론적 표현을 학습합니다.

주요 특징:
- 슬롯 어텐션을 통한 객체 중심 표현 학습
- 시맨틱 벡터 양자화를 통한 이산 표현 학습
- 오토리그레시브 프라이어를 통한 생성 모델링
- 객체의 의미론적 속성(색상, 모양 등)을 포착하는 계층적 표현

## 설치 방법

### 요구 사항

- Python 3.7+
- PyTorch 1.9+
- CUDA 지원 (선택 사항)
- Hydra 1.3+

### 설치 단계

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/svq-project.git
cd svq-project
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
svq_project/
├── src/
│   ├── models/
│   │   ├── slot_attention.py     # 슬롯 어텐션 모듈
│   │   ├── vector_quantizer.py   # 벡터 양자화 모듈
│   │   ├── transformer_decoder.py # 트랜스포머 디코더 모듈
│   │   └── svq_model.py          # SVQ 모델 구현
│   ├── data/
│   │   ├── __init__.py          # 데이터 패키지 초기화
│   │   └── datasets.py          # 데이터셋 구현
│   ├── utils/
│   │   ├── metrics.py           # 메트릭 계산 유틸리티
│   │   ├── checkpoint.py        # 체크포인트 관리
│   │   └── visualization.py     # 시각화 유틸리티
│   └── train.py                 # 학습 스크립트
├── conf/
│   ├── config.yaml              # 기본 설정 파일
│   ├── ablation/                # Ablation 실험 설정
│   └── visualization.yaml       # 시각화 설정
├── main.py                      # 메인 실행 스크립트
├── requirements.txt             # 필요한 패키지 목록
└── README.md                    # 프로젝트 설명
```

## 사용 방법

### 실행 모드

프로젝트는 다음 네 가지 실행 모드를 지원합니다:

1. **학습 (train)**: 모델 학습
   ```bash
   python main.py mode=train
   ```

2. **평가 (evaluate)**: 학습된 모델 평가
   ```bash
   python main.py mode=evaluate
   ```

3. **시각화 (visualize)**: 모델 결과 시각화
   ```bash
   python main.py mode=visualize
   ```

4. **Ablation (ablation)**: Ablation study 실행
   ```bash
   python main.py mode=ablation
   ```

### 데이터셋 선택

현재 지원하는 데이터셋:

1. **CLEVR 데이터셋**:
   ```bash
   python main.py mode=train data=clevr
   ```
   - 3D 객체 (큐브, 구, 실린더)
   - 다양한 색상과 재질
   - 설정 가능한 객체 수 (기본값: 3-6개)

2. **Shapes 데이터셋** (준비 중):
   ```bash
   python main.py mode=train data=shapes
   ```
   - 2D 도형 (원, 사각형, 삼각형)
   - 기본 색상 (빨강, 초록, 파랑)
   - 설정 가능한 객체 수

3. **MultiDsprites 데이터셋** (준비 중):
   ```bash
   python main.py mode=train data=multidsprites
   ```
   - dSprites 스타일의 2D 스프라이트
   - 다양한 색상과 모양
   - 설정 가능한 객체 수

### 설정 커스터마이징

Hydra를 통해 설정을 커스터마이징할 수 있습니다:

1. **데이터셋 설정**:
   ```bash
   # 객체 수 변경
   python main.py mode=train data=clevr data.train.num_objects=[2,4]
   
   # 이미지 크기 변경
   python main.py mode=train data=clevr data.train.image_size=64
   
   # 배치 크기 변경
   python main.py mode=train data=clevr data.train.batch_size=16
   ```

2. **학습 설정**:
   ```bash
   # 학습률 변경
   python main.py mode=train training.learning_rate=0.0005
   
   # 옵티마이저 변경
   python main.py mode=train training.optimizer_type=sgd
   
   # 스케줄러 변경
   python main.py mode=train training.scheduler_type=step
   ```

3. **시각화 설정**:
   ```bash
   # 시각화 빈도 변경
   python main.py mode=train visualization.frequency.per_epoch=2
   
   # 시각화 방법 선택
   python main.py mode=visualize visualization.embedding.methods=[pca]
   ```

### 디버그 모드

디버그 모드를 활성화하여 빠른 테스트를 수행할 수 있습니다:

```bash
python main.py mode=train debug.enabled=true
```

디버그 모드에서는:
- 더 작은 데이터셋 사용 (100 샘플)
- 더 적은 에폭 실행 (2 에폭)
- 더 작은 배치 사이즈 사용 (4)
- 상세한 로깅 활성화 (DEBUG 레벨)
- 텐서보드 로깅 활성화
- 그래디언트와 파라미터 히스토그램 시각화

### 체크포인트 관리

체크포인트는 자동으로 저장되며, 다음과 같이 관리됩니다:

1. **저장 위치**:
   - 기본: `outputs/실험이름/타임스탬프/checkpoints/`
   - 디버그 모드: `outputs/실험이름/타임스탬프/debug/checkpoints/`

2. **저장 내용**:
   - 모델 상태 (`model_state_dict`)
   - 옵티마이저 상태 (`optimizer_state_dict`)
   - 스케줄러 상태 (`scheduler_state_dict`)
   - 현재 에폭 (`epoch`)
   - 학습/검증 손실 (`train_loss`, `val_loss`)
   - 검증 메트릭 (`val_metrics`)
   - 설정 (`config`)

3. **저장 정책**:
   - 최고 성능 모델 저장 (`save_best_only=true`)
   - 주기적 저장 (`save_frequency=10`)
   - 모니터링 메트릭: `val_loss` (최소화)

### 로깅 및 시각화

1. **텐서보드 로깅**:
   ```bash
   tensorboard --logdir outputs/실험이름/타임스탬프/tensorboard
   ```
   - 학습/검증 손실
   - 메트릭 (PSNR, SSIM, 분해 점수)
   - 학습률
   - 그래디언트/파라미터 히스토그램 (디버그 모드)

2. **시각화 결과**:
   - 위치: `outputs/실험이름/타임스탬프/visualizations/`
   - 재구성 결과
   - 슬롯 분할
   - 임베딩 시각화
   - 코드북 사용 통계

## 데이터셋

이 프로젝트는 두 가지 간단한 2D 토이 데이터셋을 제공합니다:

1. **ShapesDataset**: 다양한 색상과 모양(원, 사각형, 삼각형)의 도형을 포함하는 이미지
   - 기본 도형: 원, 사각형, 삼각형
   - 기본 색상: 빨강, 초록, 파랑
   - 설정 가능한 파라미터: 이미지 크기, 객체 수, 색상, 크기 범위

2. **MultiDspritesDataset**: dSprites 스타일의 여러 스프라이트를 포함하는 이미지
   - 기본 도형: 사각형, 타원, 하트
   - 기본 색상: 빨강, 초록, 파랑, 노랑, 마젠타, 시안
   - 설정 가능한 파라미터: 이미지 크기, 객체 수, 색상, 크기 범위

데이터셋 사용 예시:

```python
from src.data import ShapesDataset, get_data_loaders

# 데이터셋 직접 사용
dataset = ShapesDataset(
    num_samples=1000,
    image_size=64,
    num_objects=4,
    object_types=['circle', 'square', 'triangle'],
    colors=['red', 'green', 'blue'],
    size_range=(0.1, 0.3)
)

# 데이터 로더 사용
train_loader, val_loader = get_data_loaders(cfg)
```

## 모델 구성 요소

### 1. 슬롯 어텐션 인코더

슬롯 어텐션 메커니즘을 사용하여 입력 이미지를 객체 중심 표현(슬롯)으로 인코딩합니다.

```python
from src.models.slot_attention import SlotAttentionEncoder

encoder = SlotAttentionEncoder(
    image_size=64,
    num_slots=4,
    num_iterations=3,
    in_channels=3,
    slot_size=64,
    hidden_dim=64
)
```

### 2. 시맨틱 벡터 양자화

슬롯 표현을 이산 코드로 양자화하여 시맨틱 표현을 학습합니다.

```python
from src.models.vector_quantizer import SemanticVectorQuantizer

quantizer = SemanticVectorQuantizer(
    num_slots=4,
    slot_dim=64,
    num_codebooks=4,
    codebook_size=512,
    code_dim=16,
    commitment_cost=0.25
)
```

### 3. 디코더

양자화된 슬롯 표현을 이미지로 디코딩합니다.

```python
from src.models.svq_model import SVQDecoder

decoder = SVQDecoder(
    slot_dim=64,
    hidden_dim=64,
    num_slots=4,
    image_size=64,
    out_channels=3
)
```

### 4. 오토리그레시브 프라이어

이산 코드의 분포를 학습하여 새로운 이미지를 생성할 수 있게 합니다.

```python
from src.models.transformer_decoder import AutoregressiveTransformer

prior = AutoregressiveTransformer(
    num_slots=4,
    num_codebooks=4,
    codebook_size=512,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    dropout=0.1
)
```

## 시각화

시각화는 `conf/visualization.yaml`에서 설정할 수 있으며, 다음 항목들을 시각화할 수 있습니다:

1. **재구성 결과**: 원본 이미지와 재구성된 이미지 비교
2. **슬롯 분할**: 각 슬롯이 담당하는 이미지 영역
3. **코드북 사용**: 코드북의 사용 빈도와 패턴
4. **임베딩**: t-SNE, UMAP, PCA를 사용한 임베딩 시각화
5. **클러스터링**: K-means, DBSCAN 등의 클러스터링 결과

시각화 실행:
```bash
python main.py visualize --config-name config
```

## Ablation Study

Ablation study는 모델의 다양한 구성 요소의 영향을 분석합니다:

1. **기본 모델 변형**:
   - 슬롯 어텐션 (반복 횟수 감소)
   - 벡터 양자화 (코드북 수 감소)
   - 오토리그레시브 프라이어 (사용/미사용)
   - 크로스 어텐션 (CNN 디코더로 대체)

2. **실행 설정**:
   - 각 실험 3회 실행 (다른 시드 사용)
   - 통계적 유의성 검정 (Wilcoxon 검정)
   - 결과 저장 (YAML, CSV, JSON)

Ablation study 실행:
```bash
python main.py ablation --config-name config
```

## 메트릭

다음 메트릭들을 계산하고 추적합니다:

1. **재구성 품질**:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)

2. **분해 품질**:
   - 분해 점수 (슬롯과 마스크 간의 상관관계)
   - 슬롯 사용 통계 (활성화율, 다양성)

3. **학습 메트릭**:
   - 손실값 (재구성, 커밋먼트, 프라이어)
   - 학습률
   - 그래디언트 통계

## 체크포인트

체크포인트 관리 기능을 통해 모델 상태를 저장하고 복원할 수 있습니다:

1. **저장 내용**:
   - 모델 상태
   - 옵티마이저 상태
   - 스케줄러 상태
   - 현재 에폭
   - 메트릭

2. **저장 주기**:
   - 최고 성능 모델
   - 주기적 저장
   - 최종 모델

## 환경별 사용 안내

### CUDA 지원

CUDA를 사용하려면 `device` 설정을 `cuda`로 지정하세요:

```bash
python main.py mode=train device=cuda
python main.py mode=train +debug.enabled=true
```

### Windows 및 Linux 호환성

이 코드는 Windows와 Linux 모두에서 테스트되었습니다:

- **Windows**: 데이터 로더의 `num_workers` 설정이 0보다 큰 경우 `if __name__ == "__main__":` 블록 내에서 코드를 실행해야 합니다.
- **Linux**: 특별한 설정 없이 실행 가능합니다.

## 참고 문헌

- Yi-Fu Wu, Minseung Lee, Sungjin Ahn. "Structured World Modeling via Semantic Vector Quantization". arXiv:2402.01203, 2024.
- Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. "Object-Centric Learning with Slot Attention". NeurIPS, 2020.
- Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. "Neural Discrete Representation Learning". NeurIPS, 2017.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
