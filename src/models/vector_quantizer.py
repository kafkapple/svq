import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    벡터 양자화 모듈
    논문: "Neural Discrete Representation Learning" (van den Oord et al., 2017)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Args:
            num_embeddings: 코드북 크기 (K)
            embedding_dim: 임베딩 차원 (D)
            commitment_cost: 커밋먼트 손실 가중치
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 코드북 초기화
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # 코드 사용 통계 (ablation 및 시각화용)
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('last_batch_usage', torch.zeros(num_embeddings))
    
    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, ..., embedding_dim]
        Returns:
            quantized: [batch_size, ..., embedding_dim]
            loss: 커밋먼트 손실
            encodings: [batch_size, ..., num_embeddings] (one-hot)
            encoding_indices: [batch_size, ...] (indices)
        """
        # 입력 형태 저장
        input_shape = inputs.shape
        
        # 평탄화
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # L2 거리 계산
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(flat_input, self.embedding.weight.t())
        
        # 가장 가까운 코드북 엔트리 찾기
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # 양자화
        quantized = self.embedding(encoding_indices)
        
        # 코드 사용 통계 업데이트 (학습 중일 때만)
        if self.training:
            batch_usage = torch.sum(encodings, dim=0)
            self.last_batch_usage = batch_usage
            self.usage_count += batch_usage
        
        # 손실 계산
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-Through Estimator
        quantized = flat_input + (quantized - flat_input).detach()
        
        # 원래 형태로 복원
        quantized = quantized.view(input_shape)
        encoding_indices = encoding_indices.view(input_shape[:-1])
        
        return quantized, loss, encodings, encoding_indices


class SemanticVectorQuantizer(nn.Module):
    """
    시맨틱 벡터 양자화 모듈 (SVQ 논문의 핵심 구현)
    """
    def __init__(
        self,
        num_slots,
        slot_dim,
        num_codebooks,
        codebook_size,
        code_dim,
        commitment_cost=0.25
    ):
        """
        Args:
            num_slots: 슬롯 수 (N)
            slot_dim: 슬롯 차원 (D)
            num_codebooks: 코드북 수 (M)
            codebook_size: 각 코드북의 크기 (K)
            code_dim: 각 코드의 차원 (d_c)
            commitment_cost: 커밋먼트 손실 가중치
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        
        assert slot_dim % num_codebooks == 0, "slot_dim must be divisible by num_codebooks"
        
        # 슬롯을 블록으로 분할하기 위한 선형 투영
        self.block_projections = nn.ModuleList([
            nn.Linear(slot_dim // num_codebooks, code_dim)
            for _ in range(num_codebooks)
        ])
        
        # 각 블록에 대한 벡터 양자화기
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, code_dim, commitment_cost)
            for _ in range(num_codebooks)
        ])
        
        # 양자화된 코드를 다시 슬롯 차원으로 투영
        self.output_projections = nn.ModuleList([
            nn.Linear(code_dim, slot_dim // num_codebooks)
            for _ in range(num_codebooks)
        ])
    
    def forward(self, slots):
        """
        Args:
            slots: [batch_size, num_slots, slot_dim]
        Returns:
            quantized_slots: [batch_size, num_slots, slot_dim]
            loss: 커밋먼트 손실
            encoding_indices: [batch_size, num_slots, num_codebooks]
        """
        batch_size, num_slots, slot_dim = slots.shape
        
        # 슬롯을 블록으로 분할
        block_size = slot_dim // self.num_codebooks
        blocks = slots.view(batch_size, num_slots, self.num_codebooks, block_size)
        
        quantized_blocks = []
        total_loss = 0
        encoding_indices_list = []
        
        # 각 블록에 대해 벡터 양자화 적용
        for i in range(self.num_codebooks):
            # 블록 투영
            block = blocks[:, :, i, :]
            projected_block = self.block_projections[i](block)
            
            # 벡터 양자화
            quantized_block, loss, _, encoding_indices = self.quantizers[i](projected_block)
            
            # 결과 투영
            output_block = self.output_projections[i](quantized_block)
            
            quantized_blocks.append(output_block)
            total_loss += loss
            encoding_indices_list.append(encoding_indices)
        
        # 양자화된 블록 결합
        quantized_slots = torch.cat(quantized_blocks, dim=2)
        
        # 인코딩 인덱스 스택
        encoding_indices = torch.stack(encoding_indices_list, dim=2)  # [batch_size, num_slots, num_codebooks]
        
        return quantized_slots, total_loss / self.num_codebooks, encoding_indices
    
    def get_codebook_usage_stats(self):
        """
        코드북 사용 통계 반환 (시각화 및 ablation용)
        """
        usage_stats = []
        for i, quantizer in enumerate(self.quantizers):
            usage_stats.append({
                'codebook_idx': i,
                'usage_count': quantizer.usage_count.cpu().numpy(),
                'last_batch_usage': quantizer.last_batch_usage.cpu().numpy()
            })
        return usage_stats
    
    def reset_usage_stats(self):
        """
        코드북 사용 통계 초기화
        """
        for quantizer in self.quantizers:
            quantizer.usage_count.zero_()
            quantizer.last_batch_usage.zero_()
    
    def get_codes_from_indices(self, indices):
        """
        인덱스로부터 코드 벡터 얻기
        Args:
            indices: [batch_size, num_slots, num_codebooks]
        Returns:
            codes: [batch_size, num_slots, num_codebooks, code_dim]
        """
        batch_size, num_slots, num_codebooks = indices.shape
        codes = []
        
        for i in range(num_codebooks):
            codebook_indices = indices[:, :, i]  # [batch_size, num_slots]
            code = self.quantizers[i].embedding(codebook_indices)  # [batch_size, num_slots, code_dim]
            codes.append(code)
        
        return torch.stack(codes, dim=2)  # [batch_size, num_slots, num_codebooks, code_dim]
