import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (PSNR) 계산"""
    mse = F.mse_loss(x, y)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def compute_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """구조적 유사성 지수(SSIM) 계산
    
    Args:
        x: 입력 이미지 [B, C, H, W]
        y: 재구성 이미지 [B, C, H, W]
        window_size: 가우시안 윈도우 크기
        
    Returns:
        torch.Tensor: SSIM 값
    """
    # 가우시안 윈도우 생성
    window = torch.tensor([
        torch.exp(-torch.tensor((x - window_size//2)**2/2, dtype=torch.float32))
        for x in range(window_size)
    ], device=x.device)
    window = window / window.sum()
    
    # 2D 가우시안 윈도우 생성
    window = window.unsqueeze(0) * window.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
    
    # 상수 정의
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    # 윈도우 적용
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=x.size(1))
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=y.size(1))
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv2d(x * x, window, padding=window_size//2, groups=x.size(1)) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=window_size//2, groups=y.size(1)) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size//2, groups=x.size(1)) - mu_xy
    
    # SSIM 계산
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return ssim_map.mean()

def compute_disentanglement_score(slots: torch.Tensor, masks: torch.Tensor) -> float:
    """슬롯 간 disentanglement 점수 계산"""
    # 슬롯 간 코사인 유사도 계산
    slots_norm = F.normalize(slots, dim=-1)
    similarity = torch.matmul(slots_norm, slots_norm.transpose(-2, -1))
    
    # 마스크 간 IoU 계산
    masks_flat = masks.view(masks.size(0), masks.size(1), -1)
    intersection = torch.matmul(masks_flat, masks_flat.transpose(-2, -1))
    union = masks_flat.sum(dim=-1, keepdim=True) + \
            masks_flat.sum(dim=-1, keepdim=True).transpose(-2, -1) - intersection
    iou = intersection / (union + 1e-8)
    
    # 유사도와 IoU의 상관관계 계산
    similarity = similarity.view(-1)
    iou = iou.view(-1)
    correlation = torch.corrcoef(torch.stack([similarity, iou]))[0, 1]
    
    # disentanglement 점수 (상관관계가 낮을수록 좋음)
    score = 1.0 - abs(correlation)
    
    return score.item()

def compute_slot_usage(slots: torch.Tensor, masks: torch.Tensor) -> Dict[str, float]:
    """슬롯 사용 통계 계산"""
    # 마스크 활성화 비율
    mask_activation = masks.mean(dim=[0, 2, 3, 4])  # [num_slots]
    
    # 슬롯 유사도
    slots_norm = F.normalize(slots, dim=-1)
    slot_similarity = torch.matmul(slots_norm, slots_norm.transpose(-2, -1))
    slot_similarity = slot_similarity.mean(dim=0)  # [num_slots, num_slots]
    
    # 슬롯 다양성 (유사도가 낮을수록 다양)
    slot_diversity = 1.0 - slot_similarity.mean(dim=-1)  # [num_slots]
    
    return {
        'mask_activation_mean': mask_activation.mean().item(),
        'mask_activation_std': mask_activation.std().item(),
        'slot_diversity_mean': slot_diversity.mean().item(),
        'slot_diversity_std': slot_diversity.std().item()
    }

def compute_metrics(x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """모든 메트릭 계산"""
    recon = outputs['recon']
    slots = outputs['slots']
    masks = outputs['masks']
    
    metrics = {
        'psnr': compute_psnr(x, recon),
        'ssim': compute_ssim(x, recon),
        'disentanglement': compute_disentanglement_score(slots, masks)
    }
    
    # 슬롯 사용 통계 추가
    slot_metrics = compute_slot_usage(slots, masks)
    metrics.update(slot_metrics)
    
    return metrics 