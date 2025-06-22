import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import requests
import zipfile
import json
import shutil
import os

logger = logging.getLogger(__name__)

class ShapesDataset(Dataset):
    """기본 도형 데이터셋"""
    def __init__(
        self,
        num_samples: int,
        image_size: int,
        num_objects: int,
        object_types: list = ['circle', 'square', 'triangle'],
        colors: list = ['red', 'green', 'blue'],
        size_range: tuple = (0.1, 0.3)
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_objects = num_objects
        self.object_types = object_types
        self.colors = colors
        self.size_range = size_range
        
        # 데이터 생성
        self.data = self._generate_data()
    
    def _generate_data(self):
        """데이터 생성"""
        # TODO: 실제 데이터 생성 로직 구현
        # 현재는 더미 데이터 반환
        return torch.randn(self.num_samples, 3, self.image_size, self.image_size)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

class MultiDspritesDataset(Dataset):
    """Multi-DSprites 데이터셋"""
    def __init__(
        self,
        num_samples: int,
        image_size: int,
        num_objects: int,
        object_types: list = ['square', 'ellipse', 'heart'],
        colors: list = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan'],
        size_range: tuple = (0.1, 0.3)
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_objects = num_objects
        self.object_types = object_types
        self.colors = colors
        self.size_range = size_range
        
        # 데이터 생성
        self.data = self._generate_data()
    
    def _generate_data(self):
        """데이터 생성"""
        # TODO: 실제 데이터 생성 로직 구현
        # 현재는 더미 데이터 반환
        return torch.randn(self.num_samples, 3, self.image_size, self.image_size)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

class CLEVRDataset(Dataset):
    """CLEVR 데이터셋"""
    def __init__(
        self,
        data_dir: str,
        split: str,
        image_size: int,
        transform: Optional[Any] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # 데이터 경로 설정
        self.images_dir = self.data_dir / split  # processed_dir 아래에 직접 있음
        self.scenes_file = self.data_dir / 'scenes' / f'CLEVR_{split}_scenes.json'
        
        # 디버깅을 위한 경로 출력
        logger.info(f"Looking for images in: {self.images_dir}")
        logger.info(f"Looking for scenes in: {self.scenes_file}")
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        # 이미지 파일 목록 로드
        self.image_files = sorted(list(self.images_dir.glob('*.png')))
        
        if not self.image_files:
            raise ValueError(
                f"No PNG images found in {self.images_dir}. "
                f"Please check if the dataset is downloaded and prepared correctly."
            )
        
        logger.info(f"Loaded {len(self.image_files)} images from {self.images_dir}")
        
        # 씬 파일 확인
        if not self.scenes_file.exists():
            logger.warning(f"Scene file not found: {self.scenes_file}")
        else:
            logger.info(f"Found scene file: {self.scenes_file}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 이미지 크기 조정
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # 이미지를 텐서로 변환
        image = torch.from_numpy(np.array(image)).float()
        image = image.permute(2, 0, 1) / 255.0  # [H, W, C] -> [C, H, W], [0, 255] -> [0, 1]
        
        # 변환 적용
        if self.transform is not None:
            image = self.transform(image)
        
        return image

def download_clevr_dataset(cfg: Dict[str, Any]) -> None:
    """CLEVR 데이터셋 다운로드 및 준비
    
    Args:
        cfg: 데이터셋 설정
    """
    data_dir = Path(cfg.data_dir)
    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    download_dir = Path(cfg.paths.download_dir)
    url = cfg.paths.url
    
    # 디렉토리 생성
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미 다운로드되어 있고 force_download이 False인 경우 스킵
    zip_path = download_dir / 'CLEVR_v1.0.zip'
    if zip_path.exists() and not cfg.force_download:
        logger.info("Dataset already downloaded. Skipping download.")
    else:
        logger.info(f"Downloading CLEVR dataset from {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    # 압축 해제
    if not (raw_dir / 'CLEVR_v1.0').exists() or cfg.force_preprocess:
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
    
    # 데이터 전처리
    if not (processed_dir / 'train').exists() or cfg.force_preprocess:
        logger.info("Preprocessing dataset...")
        clevr_dir = raw_dir / 'CLEVR_v1.0'
        
        # 학습/검증/테스트 이미지 복사
        for split in ['train', 'val', 'test']:
            src_dir = clevr_dir / 'images' / split
            dst_dir = processed_dir / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            if not src_dir.exists():
                logger.warning(f"Source directory not found: {src_dir}")
                continue
            
            # 이미지 파일 복사
            for img_file in src_dir.glob('*.png'):
                shutil.copy2(img_file, dst_dir)
            
            # 씬 정보 복사
            scenes_src = clevr_dir / 'scenes' / f'CLEVR_{split}_scenes.json'
            scenes_dst = processed_dir / 'scenes' / f'CLEVR_{split}_scenes.json'
            scenes_dst.parent.mkdir(parents=True, exist_ok=True)
            
            if scenes_src.exists():
                shutil.copy2(scenes_src, scenes_dst)
            else:
                logger.warning(f"Scene file not found: {scenes_src}")
        
        # 데이터셋 구조 확인
        train_dir = processed_dir / 'train'
        val_dir = processed_dir / 'val'
        train_images = list(train_dir.glob('*.png'))
        val_images = list(val_dir.glob('*.png'))
        
        if not train_images or not val_images:
            raise ValueError(
                f"Dataset preparation failed. Found {len(train_images)} train images "
                f"and {len(val_images)} val images. Please check the dataset structure."
            )
        
        logger.info(f"Prepared dataset with {len(train_images)} train images and {len(val_images)} val images")
    
    logger.info("Dataset preparation completed.")

def get_dataset(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """데이터셋 생성
    
    Args:
        cfg: 데이터셋 설정
        
    Returns:
        Tuple[Dataset, Dataset]: (학습 데이터셋, 검증 데이터셋)
    """
    dataset_name = cfg.name
    dataset_type = cfg.get('dataset_type', 'shapes')
    
    if dataset_name == 'clevr':
        # 데이터셋 다운로드 및 준비
        if cfg.download:
            download_clevr_dataset(cfg)
        
        # 데이터 디렉토리 확인
        data_dir = Path(cfg.data_dir)
        processed_dir = Path(cfg.paths.processed_dir)
        
        logger.info(f"Using data directory: {data_dir}")
        logger.info(f"Using processed directory: {processed_dir}")
        
        if not processed_dir.exists():
            raise ValueError(f"Processed directory not found: {processed_dir}")
        
        train_dataset = CLEVRDataset(
            data_dir=str(processed_dir),  # processed_dir 사용
            split='train',
            image_size=cfg.train.image_size,
            transform=None
        )
        val_dataset = CLEVRDataset(
            data_dir=str(processed_dir),  # processed_dir 사용
            split='val',
            image_size=cfg.val.image_size,
            transform=None
        )
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(
                f"Dataset is empty. Train samples: {len(train_dataset)}, "
                f"Val samples: {len(val_dataset)}. Please check if the dataset "
                f"is downloaded and prepared correctly at {processed_dir}"
            )
    elif dataset_name == 'custom':
        if dataset_type == 'shapes':
            train_dataset = ShapesDataset(
                num_samples=cfg.train.num_samples,
                image_size=cfg.train.image_size,
                num_objects=cfg.train.num_objects,
                object_types=cfg.train.object_types,
                colors=cfg.train.colors,
                size_range=cfg.train.size_range
            )
            val_dataset = ShapesDataset(
                num_samples=cfg.val.num_samples,
                image_size=cfg.val.image_size,
                num_objects=cfg.val.num_objects,
                object_types=cfg.val.object_types,
                colors=cfg.val.colors,
                size_range=cfg.val.size_range
            )
        elif dataset_type == 'multidsprites':
            train_dataset = MultiDspritesDataset(
                num_samples=cfg.train.num_samples,
                image_size=cfg.train.image_size,
                num_objects=cfg.train.num_objects,
                object_types=cfg.train.object_types,
                colors=cfg.train.colors,
                size_range=cfg.train.size_range
            )
            val_dataset = MultiDspritesDataset(
                num_samples=cfg.val.num_samples,
                image_size=cfg.val.image_size,
                num_objects=cfg.val.num_objects,
                object_types=cfg.val.object_types,
                colors=cfg.val.colors,
                size_range=cfg.val.size_range
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Created {dataset_name} dataset ({dataset_type})")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def get_data_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """데이터 로더 생성
    
    Args:
        cfg: 데이터셋 설정
        
    Returns:
        Tuple[DataLoader, DataLoader]: (학습 데이터 로더, 검증 데이터 로더)
    """
    train_dataset, val_dataset = get_dataset(cfg)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader 