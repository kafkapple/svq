import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models.svq_model import SVQ
from src.data.datasets import get_dataset, get_data_loaders
from src.utils.visualization import Visualizer
from src.utils.metrics import compute_metrics
from src.utils.checkpoint import CheckpointManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_debug_loaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    """디버그 모드용 데이터 로더 생성"""
    # 디버그 모드에서 사용할 샘플 수
    num_samples = min(cfg.training.debug.num_samples, len(train_dataset))
    num_val_samples = min(cfg.training.debug.num_samples // 10, len(val_dataset))
    
    # 서브셋 생성
    train_subset = Subset(train_dataset, range(num_samples))
    val_subset = Subset(val_dataset, range(num_val_samples))
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.training.debug.debug_batch_size,
        shuffle=True,
        num_workers=cfg.training.debug.debug_num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.training.debug.debug_batch_size,
        shuffle=False,
        num_workers=cfg.training.debug.debug_num_workers,
        pin_memory=True
    )
    
    logger.info(f"Debug mode: Using {num_samples} training samples and "
                f"{num_val_samples} validation samples")
    
    return train_loader, val_loader

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # 디바이스 설정
        if cfg.experiment.device == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Falling back to CPU.")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda')
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU device")
        
        self.timestamp = cfg.experiment.timestamp
        
        # 디버그 모드 설정
        if cfg.training.debug.enabled:
            logger.info("Running in debug mode")
            # 로그 레벨 설정
            logging.getLogger().setLevel(cfg.training.debug.log_level)
            # 설정 덮어쓰기
            cfg.training.num_epochs = cfg.training.debug.debug_num_epochs
            cfg.training.batch_size = cfg.training.debug.debug_batch_size
            cfg.training.num_workers = cfg.training.debug.debug_num_workers
            cfg.training.mixed_precision.enabled = cfg.training.debug.debug_mixed_precision
            cfg.training.gradient_clipping.enabled = cfg.training.debug.debug_gradient_clipping
            cfg.training.logging.tensorboard = cfg.training.debug.debug_tensorboard
            
            logger.info(f"Debug mode settings applied:")
            logger.info(f"  - num_epochs: {cfg.training.num_epochs}")
            logger.info(f"  - batch_size: {cfg.training.batch_size}")
            logger.info(f"  - num_workers: {cfg.training.num_workers}")
        
        # Set up directories
        self.setup_directories()
        
        # Set up logging
        self.setup_logging()
        
        # Set up model
        self.setup_model()
        
        # Set up optimizer and scheduler
        self.setup_optimizer()
        
        # Set up checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=cfg.training.checkpoint.save_dir,
            save_best_only=cfg.training.checkpoint.save_best_only,
            save_frequency=cfg.training.checkpoint.save_frequency,
            monitor=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode
        )
        
        # Set up visualizer
        self.visualizer = Visualizer(cfg.visualization)
        
        # Set up mixed precision training
        self.scaler = amp.GradScaler() if cfg.training.mixed_precision.enabled else None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf') if cfg.training.checkpoint.mode == 'min' else float('-inf')
        self.early_stopping_counter = 0
    
    def setup_directories(self):
        """실험 디렉토리 설정"""
        # 실험 디렉토리 설정
        if hasattr(self.cfg, 'hydra') and hasattr(self.cfg.hydra, 'run'):
            self.exp_dir = Path(self.cfg.hydra.run.dir)
        else:
            # hydra 설정이 없는 경우 기본 경로 사용
            self.exp_dir = Path('outputs') / self.cfg.experiment.name / self.cfg.experiment.timestamp
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 체크포인트 디렉토리 설정
        checkpoint_dir = self.exp_dir / self.cfg.training.checkpoint.save_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.training.checkpoint.save_dir = str(checkpoint_dir)
        
        # 설정 파일 저장
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            OmegaConf.save(config=self.cfg, f=f)
        
        # 디버그 모드일 경우 디버그 디렉토리 생성
        if self.cfg.training.debug.enabled:
            self.debug_dir = self.exp_dir / 'debug'
            self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """로깅 설정"""
        if self.cfg.training.logging.tensorboard:
            log_dir = self.exp_dir / 'tensorboard'
            if self.cfg.training.debug.enabled:
                log_dir = self.debug_dir / 'tensorboard'
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
    
    def setup_model(self):
        """모델 초기화"""
        # 모델 설정 로깅
        logger.info("Model section structure:")
        logger.info(OmegaConf.to_yaml(self.cfg.model))
        
        # SVQ 모델 초기화
        self.model = SVQ(
            image_size=self.cfg.data.train.image_size,
            in_channels=self.cfg.model.encoder.input_channels,
            num_slots=self.cfg.model.slot_attention.num_slots,
            num_iterations=self.cfg.model.slot_attention.num_iterations,
            slot_size=self.cfg.model.slot_attention.slot_dim,
            vq=self.cfg.model.vq,  # VQ 설정 전달
            hidden_dim=self.cfg.model.encoder.hidden_channels[0],
            commitment_cost=self.cfg.model.vq.commitment_cost,
            decoder_type="cnn"
        ).to(self.device)
        
        # 오토리그레시브 프라이어 초기화 (설정된 경우)
        if (self.cfg.model.ablation.use_autoregressive_prior and 
            hasattr(self.cfg.model, 'prior')):
            logger.info("Initializing autoregressive prior...")
            self.model.init_prior(
                embed_dim=self.cfg.model.prior.embed_dim,
                num_heads=self.cfg.model.prior.num_heads,
                num_layers=self.cfg.model.prior.num_layers,
                dropout=self.cfg.model.prior.dropout
            )
            logger.info("Autoregressive prior initialized successfully")
        elif self.cfg.model.ablation.use_autoregressive_prior:
            logger.warning("Autoregressive prior is enabled but prior configuration is missing. Skipping prior initialization.")
        
        # 디버그 모드일 경우 모델 정보 로깅
        if self.cfg.training.debug.enabled:
            logger.debug(f"Model architecture:\n{self.model}")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.debug(f"Total parameters: {total_params:,}")
            logger.debug(f"Trainable parameters: {trainable_params:,}")
    
    def setup_optimizer(self):
        """옵티마이저와 스케줄러 설정"""
        logger.info("Setting up optimizer and scheduler...")
        
        # 옵티마이저 설정 검증
        if not hasattr(self.cfg.training, 'optimizer'):
            raise ValueError("Optimizer configuration is missing in training settings")
        
        optimizer_cfg = self.cfg.training.optimizer
        if not hasattr(optimizer_cfg, 'name'):
            raise ValueError("Optimizer name is not specified in configuration")
        
        # 옵티마이저
        if optimizer_cfg.name.lower() == 'adam':
            logger.info("Initializing Adam optimizer...")
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                betas=tuple(optimizer_cfg.betas),
                weight_decay=optimizer_cfg.weight_decay,
                eps=optimizer_cfg.eps
            )
            logger.info(f"Adam optimizer initialized with lr={optimizer_cfg.lr}")
        elif optimizer_cfg.name.lower() == 'sgd':
            logger.info("Initializing SGD optimizer...")
            sgd_cfg = optimizer_cfg.sgd
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                momentum=sgd_cfg.momentum,
                weight_decay=optimizer_cfg.weight_decay,
                nesterov=sgd_cfg.nesterov
            )
            logger.info(f"SGD optimizer initialized with lr={optimizer_cfg.lr}")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")
        
        # 스케줄러 설정 검증
        if not hasattr(self.cfg.training, 'scheduler'):
            raise ValueError("Scheduler configuration is missing in training settings")
        
        scheduler_cfg = self.cfg.training.scheduler
        if not hasattr(scheduler_cfg, 'name'):
            raise ValueError("Scheduler name is not specified in configuration")
        
        # 스케줄러
        if scheduler_cfg.name.lower() == 'cosine':
            logger.info("Initializing CosineAnnealingLR scheduler...")
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.num_epochs,
                eta_min=scheduler_cfg.eta_min
            )
            logger.info(f"CosineAnnealingLR scheduler initialized with T_max={self.cfg.training.num_epochs}")
        elif scheduler_cfg.name.lower() == 'step':
            logger.info("Initializing StepLR scheduler...")
            step_cfg = scheduler_cfg.step
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_cfg.step_size,
                gamma=step_cfg.gamma
            )
            logger.info(f"StepLR scheduler initialized with step_size={step_cfg.step_size}")
        elif scheduler_cfg.name.lower() == 'plateau':
            logger.info("Initializing ReduceLROnPlateau scheduler...")
            plateau_cfg = scheduler_cfg.plateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=plateau_cfg.mode,
                factor=plateau_cfg.factor,
                patience=plateau_cfg.patience,
                verbose=plateau_cfg.verbose
            )
            logger.info(f"ReduceLROnPlateau scheduler initialized with patience={plateau_cfg.patience}")
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_cfg.name}")
        
        # 디버그 모드일 경우 옵티마이저 정보 로깅
        if self.cfg.training.debug.enabled:
            logger.debug(f"Optimizer: {self.optimizer}")
            logger.debug(f"Scheduler: {self.scheduler}")
            logger.debug(f"Initial learning rate: {optimizer_cfg.lr}")
            logger.debug(f"Optimizer parameters: {optimizer_cfg}")
            logger.debug(f"Scheduler parameters: {scheduler_cfg}")
            logger.debug(f"Number of epochs: {self.cfg.training.num_epochs}")
    
    def train_epoch(self, train_loader):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        loss_dict = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)
            
            # Mixed precision training
            if self.scaler is not None:
                with amp.autocast():
                    outputs = self.model(data)
                    loss, batch_loss_dict = self.model.compute_loss(
                        data,
                        outputs,
                        recon_loss_weight=self.cfg.training.loss.reconstruction_weight,
                        commitment_loss_weight=self.cfg.training.loss.commitment_weight
                    )
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                if self.cfg.training.gradient_clipping.enabled:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clipping.max_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss, batch_loss_dict = self.model.compute_loss(
                    data,
                    outputs,
                    recon_loss_weight=self.cfg.training.loss.reconstruction_weight,
                    commitment_loss_weight=self.cfg.training.loss.commitment_weight
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.cfg.training.gradient_clipping.enabled:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clipping.max_norm
                    )
                
                self.optimizer.step()
            
            # 손실 업데이트
            total_loss += loss.item()
            for k, v in batch_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = 0
                loss_dict[k] += v
            
            # 진행 상황 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                **{k: f"{v:.4f}" for k, v in batch_loss_dict.items()}
            })
            
            # 텐서보드 로깅
            if self.writer is not None and batch_idx % self.cfg.training.logging.log_frequency == 0:
                step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
                for k, v in batch_loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, step)
                
                # 디버그 모드일 경우 추가 로깅
                if self.cfg.training.debug.enabled:
                    # 그래디언트 히스토그램
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(
                                f'gradients/{name}',
                                param.grad,
                                step
                            )
                    
                    # 파라미터 히스토그램
                    for name, param in self.model.named_parameters():
                        self.writer.add_histogram(
                            f'parameters/{name}',
                            param,
                            step
                        )
        
        # 에폭 평균 손실 계산
        avg_loss = total_loss / len(train_loader)
        for k in loss_dict:
            loss_dict[k] /= len(train_loader)
        
        return avg_loss, loss_dict
    
    @torch.no_grad()
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        loss_dict = {}
        metrics_dict = {}
        
        for data in tqdm(val_loader, desc="Validation"):
            data = data.to(self.device)
            
            # Mixed precision inference
            if self.scaler is not None:
                with amp.autocast():
                    outputs = self.model(data)
                    loss, batch_loss_dict = self.model.compute_loss(
                        data,
                        outputs,
                        recon_loss_weight=self.cfg.training.loss.reconstruction_weight,
                        commitment_loss_weight=self.cfg.training.loss.commitment_weight
                    )
            else:
                outputs = self.model(data)
                loss, batch_loss_dict = self.model.compute_loss(
                    data,
                    outputs,
                    recon_loss_weight=self.cfg.training.loss.reconstruction_weight,
                    commitment_loss_weight=self.cfg.training.loss.commitment_weight
                )
            
            # 손실 업데이트
            total_loss += loss.item()
            for k, v in batch_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = 0
                loss_dict[k] += v
            
            # 메트릭 계산
            batch_metrics = compute_metrics(data, outputs)
            for k, v in batch_metrics.items():
                if k not in metrics_dict:
                    metrics_dict[k] = 0
                metrics_dict[k] += v
        
        # 평균 계산
        avg_loss = total_loss / len(val_loader)
        for k in loss_dict:
            loss_dict[k] /= len(val_loader)
        for k in metrics_dict:
            metrics_dict[k] /= len(val_loader)
        
        return avg_loss, loss_dict, metrics_dict
    
    @torch.no_grad()
    def evaluate(self):
        """모델 평가"""
        self.model.eval()
        metrics_dict = {}
        
        # 데이터 로더 설정
        _, val_dataset = get_dataset(self.cfg.data)
        if self.cfg.training.debug.enabled:
            _, val_loader = get_debug_loaders(None, val_dataset, self.cfg)
        else:
            _, val_loader = get_data_loaders(self.cfg.data)
        
        logger.info("Starting evaluation...")
        for data in tqdm(val_loader, desc="Evaluation"):
            data = data.to(self.device)
            
            # Mixed precision inference
            if self.scaler is not None:
                with amp.autocast():
                    outputs = self.model(data)
            else:
                outputs = self.model(data)
            
            # 메트릭 계산
            batch_metrics = compute_metrics(data, outputs)
            for k, v in batch_metrics.items():
                if k not in metrics_dict:
                    metrics_dict[k] = 0
                metrics_dict[k] += v
        
        # 평균 계산
        for k in metrics_dict:
            metrics_dict[k] /= len(val_loader)
        
        # 결과 로깅
        logger.info("Evaluation metrics:")
        for k, v in metrics_dict.items():
            logger.info(f"  {k}: {v:.4f}")
        
        return metrics_dict
    
    def train(self):
        """전체 학습 과정"""
        # 데이터 로더 설정
        train_dataset, val_dataset = get_dataset(self.cfg.data)
        
        # 디버그 모드일 경우 서브셋 사용
        if self.cfg.training.debug.enabled:
            train_loader, val_loader = get_debug_loaders(
                train_dataset,
                val_dataset,
                self.cfg
            )
        else:
            train_loader, val_loader = get_data_loaders(self.cfg.data)
        
        logger.info("Starting training...")
        for epoch in range(self.cfg.training.num_epochs):
            self.current_epoch = epoch
            
            # 학습
            train_loss, train_loss_dict = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_loss_dict, val_metrics = self.validate(val_loader)
            
            # 스케줄러 스텝
            self.scheduler.step()
            
            # 로깅
            logger.info(f"Epoch {epoch} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
            
            if self.writer is not None:
                self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
                self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
                for k, v in train_loss_dict.items():
                    self.writer.add_scalar(f'epoch/train_{k}', v, epoch)
                for k, v in val_loss_dict.items():
                    self.writer.add_scalar(f'epoch/val_{k}', v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'epoch/val_{k}', v, epoch)
                
                # 디버그 모드일 경우 추가 로깅
                if self.cfg.training.debug.enabled:
                    # 학습률 로깅
                    self.writer.add_scalar(
                        'learning_rate',
                        self.scheduler.get_last_lr()[0],
                        epoch
                    )
            
            # 체크포인트 저장
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': self.cfg
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            is_best = self.checkpoint_manager.save(
                checkpoint,
                val_metrics[self.cfg.training.checkpoint.monitor]
            )
            
            if is_best:
                self.best_metric = val_metrics[self.cfg.training.checkpoint.monitor]
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if (self.cfg.training.early_stopping.enabled and
                self.early_stopping_counter >= self.cfg.training.early_stopping.patience):
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # 시각화
            if epoch % self.cfg.visualization.visualization_frequency == 0:
                save_dir = self.debug_dir if self.cfg.training.debug.enabled else self.exp_dir
                self.visualizer.visualize_epoch(
                    self.model,
                    val_loader,
                    epoch,
                    save_dir / 'visualizations'
                )
        
        logger.info("Training completed!")
        
        # 최종 시각화
        save_dir = self.debug_dir if self.cfg.training.debug.enabled else self.exp_dir
        self.visualizer.visualize_final(
            self.model,
            val_loader,
            save_dir / 'visualizations'
        )
        
        if self.writer is not None:
            self.writer.close()

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """메인 함수"""
    # 시드 설정
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)
    
    # 디버그 모드 설정
    if cfg.training.debug.enabled:
        logger.info("Running in debug mode")
        logger.setLevel(cfg.training.debug.log_level)
    
    # 학습 시작
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
