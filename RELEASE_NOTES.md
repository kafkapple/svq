# SVQ (Semantic Vector-Quantized Variational Autoencoder) 프로젝트

이 프로젝트는 "Structured World Modeling via Semantic Vector Quantization" 논문의 핵심 아이디어를 PyTorch와 Hydra 프레임워크를 사용하여 구현한 것입니다.

## 프로젝트 개요

SVQ(Semantic Vector-Quantized Variational Autoencoder)는 객체 중심 학습과 벡터 양자화를 결합하여 시맨틱 신경망 이산 표현을 학습하는 모델입니다. 기존의 VQ-VAE와 달리, SVQ는 패치 수준이 아닌 객체 수준에서 의미론적 표현을 학습합니다.

## 주요 구성 요소

1. **슬롯 어텐션 인코더**: 입력 이미지를 객체 중심 표현(슬롯)으로 인코딩합니다.
2. **시맨틱 벡터 양자화**: 슬롯 표현을 이산 코드로 양자화하여 시맨틱 표현을 학습합니다.
3. **디코더**: 양자화된 슬롯 표현을 이미지로 디코딩합니다.
4. **오토리그레시브 프라이어**: 이산 코드의 분포를 학습하여 새로운 이미지를 생성합니다.

## 프로젝트 구조

```
svq_project/
├── README.md                   # 프로젝트 설명
├── requirements.txt            # 필요한 패키지 목록
├── src/                        # 소스 코드
│   ├── __init__.py             # Python 패키지 구조를 위한 파일
│   ├── train.py                # 학습 스크립트
│   ├── models/                 # 모델 구현
│   │   ├── __init__.py
│   │   ├── slot_attention.py   # 슬롯 어텐션 구현
│   │   ├── vector_quantizer.py # 벡터 양자화 구현
│   │   ├── transformer_decoder.py # 트랜스포머 디코더 구현
│   │   └── svq_model.py        # SVQ 모델 구현
│   ├── data/                   # 데이터 처리
│   │   ├── __init__.py
│   │   └── toy_datasets.py     # 토이 데이터셋 구현
│   ├── utils/                  # 유틸리티 함수
│   └── configs/                # 설정 파일
│       ├── __init__.py
│       ├── config.yaml         # 메인 설정 파일
│       └── experiment/         # 실험 설정
│           ├── __init__.py
│           └── default.yaml    # 기본 실험 설정
└── conf/                       # Hydra 설정
    ├── ablation/               # Ablation 연구 설정
    └── visualization/          # 시각화 설정
```

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

1. 학습 실행:
```bash
python src/train.py
```

2. 설정 변경하여 실행:
```bash
python src/train.py experiment.dataset.name=multidsprites experiment.model.num_slots=6
```

## 주요 수정 사항

이 프로젝트는 다음과 같은 문제들을 해결하여 정상적으로 실행되도록 수정되었습니다:

1. **Python 패키지 구조 문제 해결**
   - 누락된 `__init__.py` 파일들을 추가하여 Python이 src를 올바른 패키지로 인식하도록 했습니다.

2. **설정 파일 누락 문제 해결**
   - 누락된 `default.yaml` 설정 파일을 생성했습니다.

3. **Hydra 설정 문제 해결**
   - config.yaml에서 중복된 hydra/job_logging 항목을 제거했습니다.
   - 설정 구조와 코드 접근 방식을 일치시켰습니다.

4. **텐서 크기 불일치 문제 해결**
   - SlotAttentionEncoder의 CNN 출력과 위치 임베딩(positional embedding) 크기 불일치 문제를 해결했습니다.
   - CNN 구조를 명확히 하고 동적 크기 조정 기능을 추가했습니다.

## 데이터셋

프로젝트는 다음과 같은 토이 데이터셋을 포함합니다:

1. **ShapesDataset**: 단순한 2D 도형(원, 사각형, 삼각형 등)을 포함하는 합성 데이터셋
2. **MultiDspritesDataset**: 여러 개의 dSprites 객체를 포함하는 데이터셋

## 모델 파라미터

기본 모델 파라미터는 다음과 같습니다:

- **슬롯 수**: 4
- **슬롯 차원**: 64
- **코드북 수**: 4
- **코드북 크기**: 512
- **코드 차원**: 16
- **commitment cost**: 0.25

## 시각화 및 Ablation 연구

프로젝트는 다음과 같은 시각화 및 ablation 연구 기능을 포함합니다:

- **임베딩 시각화**: t-SNE, UMAP, PCA
- **클러스터링**: K-means, DBSCAN, 계층적 클러스터링
- **Ablation 연구**: 슬롯 어텐션 반복 횟수, 코드북 수, 오토리그레시브 프라이어, 크로스 어텐션 등

## 참고 문헌

- "Structured World Modeling via Semantic Vector Quantization" (Yi-Fu Wu, Minseung Lee, Sungjin Ahn)
- "Slot Attention" (Locatello et al., 2020)
- "Neural Discrete Representation Learning" (van den Oord et al., 2017)
