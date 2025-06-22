import os
from pathlib import Path
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CheckpointManager:
    """체크포인트 관리자"""
    
    def __init__(
        self,
        save_dir: str,
        save_best_only: bool = True,
        save_frequency: int = 10,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Args:
            save_dir: 체크포인트 저장 디렉토리
            save_best_only: 최고 성능 모델만 저장할지 여부
            save_frequency: 체크포인트 저장 주기 (에폭)
            monitor: 모니터링할 메트릭 이름
            mode: 'min' 또는 'max' (메트릭 최소화/최대화)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.monitor = monitor
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
    
    def save(self, checkpoint: Dict[str, Any], metric: float) -> bool:
        """
        체크포인트 저장
        
        Args:
            checkpoint: 저장할 체크포인트 딕셔너리
            metric: 현재 메트릭 값
        
        Returns:
            is_best: 현재 체크포인트가 최고 성능인지 여부
        """
        epoch = checkpoint['epoch']
        is_best = False
        
        # 최고 성능 체크
        if self.mode == 'min':
            is_best = metric < self.best_metric
        else:
            is_best = metric > self.best_metric
        
        if is_best:
            self.best_metric = metric
            self.best_epoch = epoch
            
            # 최고 성능 모델 저장
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved to {best_path} "
                       f"(epoch {epoch}, {self.monitor}: {metric:.4f})")
        
        # 주기적 저장
        if not self.save_best_only and epoch % self.save_frequency == 0:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 마지막 모델 저장
        last_path = self.save_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)
        
        return is_best
    
    def load_best(self, model: torch.nn.Module) -> Dict[str, Any]:
        """최고 성능 모델 로드"""
        best_path = self.save_dir / 'best_model.pth'
        if not best_path.exists():
            raise FileNotFoundError(f"No best model found at {best_path}")
        
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {best_path} "
                   f"(epoch {checkpoint['epoch']}, "
                   f"{self.monitor}: {checkpoint['val_metrics'][self.monitor]:.4f})")
        
        return checkpoint
    
    def load_last(self, model: torch.nn.Module) -> Dict[str, Any]:
        """마지막 모델 로드"""
        last_path = self.save_dir / 'last_model.pth'
        if not last_path.exists():
            raise FileNotFoundError(f"No last model found at {last_path}")
        
        checkpoint = torch.load(last_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded last model from {last_path} "
                   f"(epoch {checkpoint['epoch']})")
        
        return checkpoint
    
    def load_checkpoint(self, model: torch.nn.Module, epoch: int) -> Dict[str, Any]:
        """특정 에폭의 체크포인트 로드"""
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path} "
                   f"(epoch {epoch})")
        
        return checkpoint 