import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


class ShapesDataset(data.Dataset):
    """
    간단한 2D 도형 데이터셋 생성
    다양한 색상과 모양의 도형을 포함하는 이미지 생성
    """
    def __init__(self, 
                 num_samples=10000, 
                 image_size=64, 
                 max_shapes=4,
                 transform=None,
                 seed=42):
        """
        Args:
            num_samples: 데이터셋 크기
            image_size: 이미지 크기
            max_shapes: 이미지당 최대 도형 수
            transform: 추가 변환
            seed: 랜덤 시드
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_shapes = max_shapes
        self.transform = transform
        
        # 랜덤 시드 설정
        np.random.seed(seed)
        
        # 가능한 도형 종류
        self.shapes = ['circle', 'square', 'triangle']
        
        # 가능한 색상 (RGB)
        self.colors = [
            [255, 0, 0],    # 빨강
            [0, 255, 0],    # 초록
            [0, 0, 255],    # 파랑
            [255, 255, 0],  # 노랑
            [255, 0, 255],  # 마젠타
            [0, 255, 255],  # 시안
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 빈 이미지 생성 (검은색 배경)
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # 이미지에 포함될 도형 수 결정 (1 ~ max_shapes)
        num_shapes = np.random.randint(1, self.max_shapes + 1)
        
        # 각 도형에 대해
        for _ in range(num_shapes):
            # 도형 종류 선택
            shape_type = np.random.choice(self.shapes)
            
            # 색상 선택
            color = self.colors[np.random.randint(0, len(self.colors))]
            
            # 크기 결정 (이미지 크기의 10% ~ 30%)
            size = np.random.randint(int(self.image_size * 0.1), int(self.image_size * 0.3))
            
            # 위치 결정 (도형이 이미지 안에 완전히 들어가도록)
            x = np.random.randint(size, self.image_size - size)
            y = np.random.randint(size, self.image_size - size)
            
            # 도형 그리기
            if shape_type == 'circle':
                self._draw_circle(image, x, y, size, color)
            elif shape_type == 'square':
                self._draw_square(image, x, y, size, color)
            elif shape_type == 'triangle':
                self._draw_triangle(image, x, y, size, color)
        
        # numpy 배열을 PIL 이미지로 변환
        image = Image.fromarray(image)
        
        # 추가 변환 적용
        if self.transform:
            image = self.transform(image)
        else:
            # 기본 변환 (ToTensor)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        return image
    
    def _draw_circle(self, image, x, y, size, color):
        """원 그리기"""
        for i in range(self.image_size):
            for j in range(self.image_size):
                # 원의 방정식: (i-x)^2 + (j-y)^2 < r^2
                if (i - x) ** 2 + (j - y) ** 2 < (size // 2) ** 2:
                    image[j, i] = color
    
    def _draw_square(self, image, x, y, size, color):
        """정사각형 그리기"""
        half_size = size // 2
        x_min = max(0, x - half_size)
        x_max = min(self.image_size, x + half_size)
        y_min = max(0, y - half_size)
        y_max = min(self.image_size, y + half_size)
        
        image[y_min:y_max, x_min:x_max] = color
    
    def _draw_triangle(self, image, x, y, size, color):
        """삼각형 그리기"""
        half_size = size // 2
        
        # 삼각형의 세 꼭지점
        x1, y1 = x, y - half_size  # 상단
        x2, y2 = x - half_size, y + half_size  # 좌하단
        x3, y3 = x + half_size, y + half_size  # 우하단
        
        # 삼각형 내부의 모든 픽셀 확인
        for i in range(max(0, x - half_size), min(self.image_size, x + half_size + 1)):
            for j in range(max(0, y - half_size), min(self.image_size, y + half_size + 1)):
                # 점 (i,j)가 삼각형 내부에 있는지 확인 (무게중심 좌표 이용)
                if self._point_in_triangle(i, j, x1, y1, x2, y2, x3, y3):
                    image[j, i] = color
    
    def _point_in_triangle(self, x, y, x1, y1, x2, y2, x3, y3):
        """점 (x,y)가 삼각형 내부에 있는지 확인"""
        def sign(p1x, p1y, p2x, p2y, p3x, p3y):
            return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
        
        d1 = sign(x, y, x1, y1, x2, y2)
        d2 = sign(x, y, x2, y2, x3, y3)
        d3 = sign(x, y, x3, y3, x1, y1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        # 모든 부호가 같으면 삼각형 내부
        return not (has_neg and has_pos)
    
    def visualize_samples(self, num_samples=5, save_path=None):
        """데이터셋 샘플 시각화"""
        plt.figure(figsize=(15, 3))
        
        for i in range(num_samples):
            sample = self[np.random.randint(0, len(self))]
            
            # 텐서를 이미지로 변환
            if isinstance(sample, torch.Tensor):
                sample = sample.permute(1, 2, 0).numpy()
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(sample)
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class MultiDspritesDataset(data.Dataset):
    """
    Multi-dSprites 데이터셋
    dSprites 데이터셋을 기반으로 여러 스프라이트를 포함하는 이미지 생성
    """
    def __init__(self, 
                 num_samples=10000, 
                 image_size=64, 
                 max_sprites=4,
                 transform=None,
                 seed=42):
        """
        Args:
            num_samples: 데이터셋 크기
            image_size: 이미지 크기
            max_sprites: 이미지당 최대 스프라이트 수
            transform: 추가 변환
            seed: 랜덤 시드
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_sprites = max_sprites
        self.transform = transform
        
        # 랜덤 시드 설정
        np.random.seed(seed)
        
        # 스프라이트 종류 (dSprites에서는 3가지: 정사각형, 타원, 하트)
        self.sprite_types = ['square', 'ellipse', 'heart']
        
        # 가능한 색상 (RGB)
        self.colors = [
            [255, 0, 0],    # 빨강
            [0, 255, 0],    # 초록
            [0, 0, 255],    # 파랑
            [255, 255, 0],  # 노랑
            [255, 0, 255],  # 마젠타
            [0, 255, 255],  # 시안
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 빈 이미지 생성 (흰색 배경)
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
        
        # 이미지에 포함될 스프라이트 수 결정 (1 ~ max_sprites)
        num_sprites = np.random.randint(1, self.max_sprites + 1)
        
        # 각 스프라이트에 대해
        for _ in range(num_sprites):
            # 스프라이트 종류 선택
            sprite_type = np.random.choice(self.sprite_types)
            
            # 색상 선택
            color = self.colors[np.random.randint(0, len(self.colors))]
            
            # 크기 결정 (이미지 크기의 10% ~ 25%)
            size = np.random.randint(int(self.image_size * 0.1), int(self.image_size * 0.25))
            
            # 위치 결정 (스프라이트가 이미지 안에 완전히 들어가도록)
            x = np.random.randint(size, self.image_size - size)
            y = np.random.randint(size, self.image_size - size)
            
            # 회전 각도 결정 (0 ~ 360도)
            angle = np.random.randint(0, 360)
            
            # 스프라이트 그리기
            if sprite_type == 'square':
                self._draw_square(image, x, y, size, color, angle)
            elif sprite_type == 'ellipse':
                self._draw_ellipse(image, x, y, size, color, angle)
            elif sprite_type == 'heart':
                self._draw_heart(image, x, y, size, color, angle)
        
        # numpy 배열을 PIL 이미지로 변환
        image = Image.fromarray(image)
        
        # 추가 변환 적용
        if self.transform:
            image = self.transform(image)
        else:
            # 기본 변환 (ToTensor)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        return image
    
    def _draw_square(self, image, x, y, size, color, angle):
        """회전된 정사각형 그리기"""
        # 간단한 구현을 위해 회전 없이 그리기
        half_size = size // 2
        x_min = max(0, x - half_size)
        x_max = min(self.image_size, x + half_size)
        y_min = max(0, y - half_size)
        y_max = min(self.image_size, y + half_size)
        
        image[y_min:y_max, x_min:x_max] = color
    
    def _draw_ellipse(self, image, x, y, size, color, angle):
        """타원 그리기"""
        # 타원의 가로/세로 반지름
        a = size // 2
        b = size // 3
        
        # 간단한 구현을 위해 회전 없이 그리기
        for i in range(self.image_size):
            for j in range(self.image_size):
                # 타원의 방정식: (i-x)^2/a^2 + (j-y)^2/b^2 < 1
                if ((i - x) ** 2) / (a ** 2) + ((j - y) ** 2) / (b ** 2) < 1:
                    image[j, i] = color
    
    def _draw_heart(self, image, x, y, size, color, angle):
        """하트 모양 그리기 (간단한 근사)"""
        # 간단한 구현을 위해 원 두 개와 삼각형으로 근사
        r = size // 4
        
        # 두 원의 중심
        x1 = x - r
        y1 = y - r
        x2 = x + r
        y2 = y - r
        
        # 삼각형의 세 꼭지점
        tx1, ty1 = x - size//2, y - r
        tx2, ty2 = x + size//2, y - r
        tx3, ty3 = x, y + size//2
        
        # 두 원 그리기
        for i in range(self.image_size):
            for j in range(self.image_size):
                if ((i - x1) ** 2 + (j - y1) ** 2 < r ** 2) or ((i - x2) ** 2 + (j - y2) ** 2 < r ** 2):
                    image[j, i] = color
        
        # 삼각형 그리기
        for i in range(max(0, x - size//2), min(self.image_size, x + size//2 + 1)):
            for j in range(max(0, y - r), min(self.image_size, y + size//2 + 1)):
                if self._point_in_triangle(i, j, tx1, ty1, tx2, ty2, tx3, ty3):
                    image[j, i] = color
    
    def _point_in_triangle(self, x, y, x1, y1, x2, y2, x3, y3):
        """점 (x,y)가 삼각형 내부에 있는지 확인"""
        def sign(p1x, p1y, p2x, p2y, p3x, p3y):
            return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
        
        d1 = sign(x, y, x1, y1, x2, y2)
        d2 = sign(x, y, x2, y2, x3, y3)
        d3 = sign(x, y, x3, y3, x1, y1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        # 모든 부호가 같으면 삼각형 내부
        return not (has_neg and has_pos)
    
    def visualize_samples(self, num_samples=5, save_path=None):
        """데이터셋 샘플 시각화"""
        plt.figure(figsize=(15, 3))
        
        for i in range(num_samples):
            sample = self[np.random.randint(0, len(self))]
            
            # 텐서를 이미지로 변환
            if isinstance(sample, torch.Tensor):
                sample = sample.permute(1, 2, 0).numpy()
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(sample)
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def get_dataset(dataset_name, **kwargs):
    """
    데이터셋 이름에 따라 적절한 데이터셋 반환
    """
    if dataset_name == 'shapes':
        return ShapesDataset(**kwargs)
    elif dataset_name == 'multi_dsprites':
        return MultiDspritesDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_data_loaders(dataset_name, batch_size=32, num_workers=4, train_ratio=0.8, **kwargs):
    """
    데이터셋 이름에 따라 학습/검증 데이터 로더 반환
    """
    # 데이터셋 생성
    dataset = get_dataset(dataset_name, **kwargs)
    
    # 학습/검증 분할
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 데이터 로더 생성
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
