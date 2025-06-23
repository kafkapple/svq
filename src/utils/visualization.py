import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

class Visualizer:
    """시각화 클래스"""
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Args:
            cfg: 시각화 설정
        """
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 스타일 설정
        sns.set_style(cfg.style)  # seaborn 스타일 설정
        self.colormap = cfg.colormap
    
    def visualize_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        save_dir: Path
    ):
        """에폭별 시각화"""
        model.eval()
        device = next(model.parameters()).device
        
        # 배치 데이터 가져오기
        batch = next(iter(dataloader))
        images = batch.to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        # 재구성 시각화
        self._visualize_reconstruction(
            images, outputs, epoch, save_dir / 'reconstruction'
        )
        
        # 슬롯 시각화
        self._visualize_slots(
            outputs['slots'], outputs['masks'], epoch, save_dir / 'slots'
        )
        
        # 코드북 시각화
        self._visualize_codebook(
            model, outputs['encoding_indices'], epoch, save_dir / 'codebook'
        )
        
        # 임베딩 시각화
        self._visualize_embeddings(
            outputs['slots'], outputs['quantized_slots'], epoch, save_dir / 'embeddings'
        )
        
        # 클러스터링 시각화
        self._visualize_clustering(
            outputs['slots'], outputs['masks'], epoch, save_dir / 'clustering'
        )
    
    def visualize_final(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        save_dir: Path
    ):
        """최종 시각화"""
        model.eval()
        device = next(model.parameters()).device
        
        # 전체 데이터셋에 대한 메트릭 계산
        all_metrics = []
        all_slots = []
        all_masks = []
        all_indices = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch.to(device)
                outputs = model(images)
                
                # 메트릭 계산
                metrics = compute_metrics(images, outputs)
                all_metrics.append(metrics)
                
                # 데이터 수집
                all_slots.append(outputs['slots'].cpu())
                all_masks.append(outputs['masks'].cpu())
                all_indices.append(outputs['encoding_indices'].cpu())
        
        # 메트릭 평균 계산
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        
        # 데이터 결합
        all_slots = torch.cat(all_slots, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        # 최종 시각화
        self._visualize_metrics(avg_metrics, save_dir / 'metrics')
        self._visualize_embeddings(
            all_slots, None, 'final', save_dir / 'embeddings'
        )
        self._visualize_clustering(
            all_slots, all_masks, 'final', save_dir / 'clustering'
        )
        self._visualize_codebook_usage(
            all_indices, save_dir / 'codebook'
        )
    
    def _visualize_reconstruction(
        self,
        images: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        epoch: int,
        save_dir: Path
    ):
        """재구성 시각화"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 원본과 재구성 이미지 시각화
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(4):
            # 원본 이미지
            axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
            axes[0, i].set_title(f'Original {i}')
            axes[0, i].axis('off')
            
            # 재구성 이미지
            axes[1, i].imshow(outputs['recon'][i].cpu().permute(1, 2, 0))
            axes[1, i].set_title(f'Reconstructed {i}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'reconstruction_epoch_{epoch}.png')
        plt.close()
    
    def _visualize_slots(
        self,
        slots: torch.Tensor,
        masks: torch.Tensor,
        epoch: int,
        save_dir: Path
    ):
        """슬롯 시각화"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 슬롯 어텐션 맵 시각화
        fig, axes = plt.subplots(4, slots.size(1), figsize=(4*slots.size(1), 16))
        
        for i in range(4):  # 배치 샘플
            for j in range(slots.size(1)):  # 슬롯
                # 어텐션 맵
                attn_map = masks[i, j, 0].cpu()
                axes[i, j].imshow(attn_map, cmap=self.colormap)
                axes[i, j].set_title(f'Slot {j}')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'slots_epoch_{epoch}.png')
        plt.close()
        
        # 슬롯 임베딩 시각화
        slots_norm = F.normalize(slots, dim=-1)
        similarity = torch.matmul(slots_norm, slots_norm.transpose(-2, -1))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity[0].cpu(),
            cmap=self.colormap,
            vmin=-1,
            vmax=1,
            center=0,
            square=True
        )
        plt.title('Slot Similarity Matrix')
        plt.savefig(save_dir / f'slot_similarity_epoch_{epoch}.png')
        plt.close()
    
    def _visualize_codebook(
        self,
        model: torch.nn.Module,
        encoding_indices: torch.Tensor,
        epoch: int,
        save_dir: Path
    ):
        """코드북 시각화"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 코드북 사용 통계 (list of dicts)
        usage_stats = model.get_codebook_usage_stats()

        # 통계에서 usage_count만 추출하여 배열로 변환
        usage_counts = []
        for stat in usage_stats:
            count = stat.get('usage_count')
            if isinstance(count, torch.Tensor):
                count = count.detach().cpu().numpy()
            usage_counts.append(count)
        usage_counts = np.asarray(usage_counts)

        # 사용 히스토그램
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(usage_counts)), usage_counts)
        plt.title('Codebook Usage Distribution')
        plt.xlabel('Code Index')
        plt.ylabel('Usage Count')
        plt.savefig(save_dir / f'codebook_usage_epoch_{epoch}.png')
        plt.close()
        
        # 인코딩 인덱스 히트맵
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            encoding_indices[0].cpu(),
            cmap=self.colormap,
            square=True
        )
        plt.title('Encoding Indices Heatmap')
        plt.savefig(save_dir / f'encoding_indices_epoch_{epoch}.png')
        plt.close()
    
    def _visualize_embeddings(
        self,
        slots: torch.Tensor,
        quantized_slots: Optional[torch.Tensor],
        epoch: int,
        save_dir: Path
    ):
        """임베딩 시각화"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 슬롯 임베딩을 2D로 축소
        slots_flat = slots.view(-1, slots.size(-1)).cpu()
        
        for method in self.cfg.embedding.methods:
            if method == 'tsne':
                reducer = TSNE(
                    n_components=2,
                    perplexity=self.cfg.embedding.perplexity,
                    random_state=self.cfg.embedding.random_state
                )
            elif method == 'umap':
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=self.cfg.embedding.n_neighbors,
                    min_dist=self.cfg.embedding.min_dist,
                    random_state=self.cfg.embedding.random_state
                )
            elif method == 'pca':
                reducer = PCA(n_components=2)
            else:
                continue
            
            # 임베딩 축소
            embeddings_2d = reducer.fit_transform(slots_flat)
            
            # 시각화
            plt.figure(figsize=(10, 8))
            plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=np.arange(len(embeddings_2d)),
                cmap=self.colormap
            )
            plt.title(f'Slot Embeddings ({method.upper()})')
            plt.savefig(save_dir / f'embeddings_{method}_epoch_{epoch}.png')
            plt.close()
            
            # 양자화된 슬롯이 있는 경우
            if quantized_slots is not None:
                quantized_flat = quantized_slots.view(-1, quantized_slots.size(-1)).cpu()
                quantized_2d = reducer.fit_transform(quantized_flat)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    quantized_2d[:, 0],
                    quantized_2d[:, 1],
                    c=np.arange(len(quantized_2d)),
                    cmap=self.colormap
                )
                plt.title(f'Quantized Slot Embeddings ({method.upper()})')
                plt.savefig(save_dir / f'quantized_embeddings_{method}_epoch_{epoch}.png')
                plt.close()
    
    def _visualize_clustering(
        self,
        slots: torch.Tensor,
        masks: torch.Tensor,
        epoch: int,
        save_dir: Path
    ):
        """클러스터링 시각화"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 슬롯 임베딩을 2D로 축소
        slots_flat = slots.view(-1, slots.size(-1)).cpu()
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(slots_flat)
        
        for method in self.cfg.clustering.methods:
            if method == 'kmeans':
                for n_clusters in self.cfg.clustering.n_clusters:
                    # K-means 클러스터링
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(slots_flat)
                    
                    # 실루엣 점수 계산
                    silhouette = silhouette_score(slots_flat, labels)
                    
                    # 시각화
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(
                        embeddings_2d[:, 0],
                        embeddings_2d[:, 1],
                        c=labels,
                        cmap=self.colormap
                    )
                    plt.colorbar(scatter)
                    plt.title(f'K-means Clustering (k={n_clusters}, '
                             f'silhouette={silhouette:.3f})')
                    plt.savefig(save_dir / f'kmeans_k{n_clusters}_epoch_{epoch}.png')
                    plt.close()
            
            elif method == 'dbscan':
                # DBSCAN 클러스터링
                dbscan = DBSCAN(
                    eps=self.cfg.clustering.eps,
                    min_samples=self.cfg.clustering.min_samples
                )
                labels = dbscan.fit_predict(slots_flat)
                
                # 시각화
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(
                    embeddings_2d[:, 0],
                    embeddings_2d[:, 1],
                    c=labels,
                    cmap=self.colormap
                )
                plt.colorbar(scatter)
                plt.title('DBSCAN Clustering')
                plt.savefig(save_dir / f'dbscan_epoch_{epoch}.png')
                plt.close()
    
    def _visualize_metrics(
        self,
        metrics: Dict[str, float],
        save_dir: Path
    ):
        """메트릭 시각화"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 바 차트
        plt.figure(figsize=(12, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.xticks(rotation=45)
        plt.title('Model Metrics')
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_bar.png')
        plt.close()
        
        # 메트릭 테이블
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[list(metrics.keys()), list(metrics.values())],
                fill_color='lavender',
                align='left'
            )
        )])
        fig.write_html(save_dir / 'metrics_table.html')
    
    def _visualize_codebook_usage(
        self,
        encoding_indices: torch.Tensor,
        save_dir: Path
    ):
        """코드북 사용 시각화"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 코드북 사용 통계
        usage = torch.zeros(encoding_indices.max().item() + 1)
        for indices in encoding_indices:
            usage[indices] += 1
        
        # 사용 분포 시각화
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(usage)), usage.cpu())
        plt.title('Codebook Usage Distribution')
        plt.xlabel('Code Index')
        plt.ylabel('Usage Count')
        plt.savefig(save_dir / 'codebook_usage_final.png')
        plt.close()
        
        # 사용 히트맵
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            encoding_indices[0].cpu(),
            cmap=self.colormap,
            square=True
        )
        plt.title('Encoding Indices Heatmap')
        plt.savefig(save_dir / 'encoding_indices_final.png')
        plt.close() 