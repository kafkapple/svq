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
│   │   └── toy_datasets.py       # 토이 데이터셋 구현
│   ├── utils/
│   │   └── visualization.py      # 시각화 유틸리티
│   ├── configs/
│   │   ├── config.yaml           # 기본 설정 파일
│   │   └── experiment/           # 실험별 설정 파일
│   └── train.py                  # 학습 스크립트
├── conf/
│   ├── ablation/                 # Ablation 실험 설정
│   └── visualization/            # 시각화 설정
├── requirements.txt              # 필요한 패키지 목록
└── README.md                     # 프로젝트 설명
```

## 사용 방법

### 학습

기본 설정으로 모델을 학습하려면:

```bash
python -m src.train
```

특정 설정 파일을 사용하려면:

```bash
python -m src.train --config-name=experiment/my_experiment
```

하이퍼파라미터를 명령줄에서 직접 변경하려면:

```bash
python -m src.train model.num_slots=6 training.learning_rate=0.0002
```

### 시각화

학습 중 시각화는 TensorBoard를 통해 확인할 수 있습니다:

```bash
tensorboard --logdir=logs
```

## 토이 데이터셋

이 프로젝트는 두 가지 간단한 2D 토이 데이터셋을 제공합니다:

1. **ShapesDataset**: 다양한 색상과 모양(원, 사각형, 삼각형)의 도형을 포함하는 이미지
2. **MultiDspritesDataset**: dSprites 스타일의 여러 스프라이트를 포함하는 이미지

데이터셋 예시:

```python
from src.data.toy_datasets import ShapesDataset

# 데이터셋 생성
dataset = ShapesDataset(num_samples=1000, image_size=64, max_shapes=4)

# 샘플 시각화
dataset.visualize_samples(num_samples=5, save_path="samples.png")
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

## Ablation Study

모델의 다양한 구성 요소에 대한 ablation study를 실행할 수 있습니다:

```bash
python -m src.train ablation.enabled=true
```

기본적으로 다음 구성 요소에 대한 ablation이 수행됩니다:
- 슬롯 어텐션 (반복 횟수 감소)
- 벡터 양자화 (코드북 수 감소)
- 오토리그레시브 프라이어 (사용/미사용)
- 크로스 어텐션 (CNN 디코더로 대체)

## 환경별 사용 안내

### CUDA 지원

CUDA를 사용하려면 `device` 설정을 `cuda`로 지정하세요:

```bash
python -m src.train device=cuda
```

CUDA가 사용 가능한지 확인하려면:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
```

### Windows 및 Linux 호환성

이 코드는 Windows와 Linux 모두에서 테스트되었습니다. 다음 사항에 주의하세요:

- **Windows**: 데이터 로더의 `num_workers` 설정이 0보다 큰 경우 `if __name__ == "__main__":` 블록 내에서 코드를 실행해야 합니다.
- **Linux**: 특별한 설정 없이 실행 가능합니다.

## 결과 해석

### 학습 결과

학습 중 다음과 같은 결과를 확인할 수 있습니다:

1. **재구성 이미지**: 모델이 입력 이미지를 얼마나 잘 재구성하는지 보여줍니다.
2. **슬롯 분할**: 각 슬롯이 이미지의 어떤 부분을 담당하는지 보여주는 마스크입니다.
3. **코드북 사용 통계**: 각 코드북의 코드가 얼마나 자주 사용되는지 보여줍니다.

### 생성 결과

오토리그레시브 프라이어를 학습한 후에는 새로운 이미지를 생성할 수 있습니다:

```python
# 모델 로드
model = SVQ(...)
model.load_state_dict(torch.load("checkpoints/best_model.pth")["model_state_dict"])
model.init_prior(...)

# 이미지 생성
generated_images = model.generate(batch_size=8, temperature=1.0)
```

## 예제 코드

### 전체 학습 예제

```python
from src.models.svq_model import SVQ
from src.data.toy_datasets import get_data_loaders
import torch
import torch.optim as optim

# 데이터 로더 생성
train_loader, val_loader = get_data_loaders(
    dataset_name='shapes',
    batch_size=32,
    num_workers=4,
    num_samples=10000,
    image_size=64,
    max_shapes=4
)

# 모델 생성
model = SVQ(
    image_size=64,
    in_channels=3,
    num_slots=4,
    num_iterations=3,
    slot_size=64,
    num_codebooks=4,
    codebook_size=512,
    code_dim=16,
    hidden_dim=64,
    commitment_cost=0.25
)

# 오토리그레시브 프라이어 초기화 (선택적)
model.init_prior(
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    dropout=0.1
)

# 옵티마이저 설정
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 학습 루프
for epoch in range(100):
    for batch_idx, data in enumerate(train_loader):
        # 모델 순전파
        outputs = model(data)
        
        # 손실 계산
        loss, loss_dict = model.compute_loss(
            data,
            outputs,
            recon_loss_weight=1.0,
            commitment_loss_weight=0.25
        )
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 시각화 예제

```python
import matplotlib.pyplot as plt
import torch

# 모델 로드
model = SVQ(...)
model.load_state_dict(torch.load("checkpoints/best_model.pth")["model_state_dict"])

# 샘플 데이터
data = next(iter(val_loader))

# 모델 순전파
outputs = model(data)

# 결과 시각화
plt.figure(figsize=(15, 5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(data[0].permute(1, 2, 0).numpy())
plt.title("Input")
plt.axis("off")

# 재구성 이미지
plt.subplot(1, 3, 2)
plt.imshow(outputs["recon"][0].detach().permute(1, 2, 0).numpy())
plt.title("Reconstruction")
plt.axis("off")

# 첫 번째 슬롯 마스크
plt.subplot(1, 3, 3)
plt.imshow(outputs["masks"][0, 0, 0].detach().numpy(), cmap="viridis")
plt.title("Slot 1 Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig("visualization.png")
```

## 참고 문헌

- Yi-Fu Wu, Minseung Lee, Sungjin Ahn. "Structured World Modeling via Semantic Vector Quantization". arXiv:2402.01203, 2024.
- Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. "Object-Centric Learning with Slot Attention". NeurIPS, 2020.
- Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. "Neural Discrete Representation Learning". NeurIPS, 2017.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
