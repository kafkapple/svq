import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.slot_attention import SlotAttentionEncoder
from src.models.vector_quantizer import SemanticVectorQuantizer
from src.models.transformer_decoder import AutoregressiveTransformer


class SVQDecoder(nn.Module):
    """
    SVQ 디코더 모듈
    양자화된 슬롯 표현을 이미지로 디코딩
    """
    def __init__(
        self,
        slot_dim,
        hidden_dim=64,
        num_slots=4,
        image_size=64,
        out_channels=3
    ):
        """
        Args:
            slot_dim: 슬롯 차원
            hidden_dim: 은닉층 차원
            num_slots: 슬롯 수
            image_size: 출력 이미지 크기
            out_channels: 출력 채널 수
        """
        super().__init__()
        
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.image_size = image_size
        
        # 슬롯을 디코더 입력으로 변환하는 MLP
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 위치 인코딩 생성
        self.pos_embed = nn.Parameter(torch.randn(1, hidden_dim, image_size//8, image_size//8))
        
        # 디코더 CNN
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels + 1, kernel_size=5, padding=2)
        )
    
    def forward(self, slots):
        """
        Args:
            slots: [batch_size, num_slots, slot_dim]
        Returns:
            recon: [batch_size, out_channels, image_size, image_size]
            masks: [batch_size, num_slots, 1, image_size, image_size]
            slots_recon: [batch_size, num_slots, out_channels, image_size, image_size]
        """
        batch_size, num_slots, slot_dim = slots.shape
        
        # 슬롯 투영
        slots = self.slot_proj(slots)  # [batch_size, num_slots, hidden_dim]
        
        # 각 슬롯을 개별적으로 디코딩
        slots_recon = []
        for slot_idx in range(num_slots):
            slot = slots[:, slot_idx]  # [batch_size, hidden_dim]
            
            # 슬롯을 공간적 특성 맵으로 변환
            h = w = self.image_size // 8
            slot = slot.view(batch_size, self.hidden_dim, 1, 1).expand(-1, -1, h, w)
            
            # 위치 임베딩 추가
            slot = slot + self.pos_embed
            
            # CNN 디코더 적용
            slot_recon = self.decoder_cnn(slot)  # [batch_size, out_channels+1, image_size, image_size]
            
            slots_recon.append(slot_recon)
        
        # 모든 슬롯의 재구성 결과 스택
        slots_recon = torch.stack(slots_recon, dim=1)  # [batch_size, num_slots, out_channels+1, image_size, image_size]
        
        # 마스크와 재구성 분리
        masks = slots_recon[:, :, :1]  # [batch_size, num_slots, 1, image_size, image_size]
        recons = slots_recon[:, :, 1:]  # [batch_size, num_slots, out_channels, image_size, image_size]
        
        # 마스크에 소프트맥스 적용
        masks = F.softmax(masks, dim=1)
        
        # 마스크와 재구성 결합
        recon = torch.sum(masks * recons, dim=1)  # [batch_size, out_channels, image_size, image_size]
        
        return recon, masks, recons


class SVQ(nn.Module):
    """
    Semantic Vector-Quantized Variational Autoencoder (SVQ)
    논문: "Structured World Modeling via Semantic Vector Quantization"
    """
    def __init__(
        self,
        image_size=64,
        in_channels=3,
        num_slots=4,
        num_iterations=3,
        slot_size=64,
        vq=None,  # VQ 설정을 받도록 수정
        hidden_dim=64,
        commitment_cost=0.25,
        decoder_type="cnn"
    ):
        """
        Args:
            image_size: 입력 이미지 크기
            in_channels: 입력 채널 수
            num_slots: 슬롯 수 (N)
            num_iterations: 슬롯 어텐션 반복 횟수
            slot_size: 슬롯 차원 (D)
            vq: VQ 설정 딕셔너리
            hidden_dim: 은닉층 차원
            commitment_cost: 커밋먼트 손실 가중치
            decoder_type: 디코더 타입 ("cnn" 또는 "transformer")
        """
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_size = slot_size
        self.hidden_dim = hidden_dim
        self.commitment_cost = commitment_cost
        self.decoder_type = decoder_type
        
        # VQ 설정 검증
        if vq is None:
            raise ValueError("VQ 설정이 필요합니다.")
        
        self.num_codebooks = vq.get('num_codebooks', 128)
        self.codebook_size = vq.get('num_embeddings', 512)
        self.code_dim = vq.get('embedding_dim', 128)
        
        # 슬롯 어텐션 인코더
        self.encoder = SlotAttentionEncoder(
            image_size=image_size,
            num_slots=num_slots,
            num_iterations=num_iterations,
            in_channels=in_channels,
            slot_size=slot_size,
            hidden_dim=hidden_dim
        )
        
        # 시맨틱 벡터 양자화
        self.quantizer = SemanticVectorQuantizer(
            num_slots=num_slots,
            slot_dim=slot_size,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            code_dim=self.code_dim,
            commitment_cost=commitment_cost
        )
        
        # 디코더
        self.decoder = SVQDecoder(
            slot_dim=slot_size,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            image_size=image_size,
            out_channels=in_channels
        )
        
        # 오토리그레시브 프라이어 (선택적)
        self.prior = None
    
    def init_prior(self, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        """
        오토리그레시브 프라이어 초기화 (선택적)
        """
        self.prior = AutoregressiveTransformer(
            num_slots=self.num_slots,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def encode(self, x):
        """
        인코딩 함수
        Args:
            x: [batch_size, in_channels, image_size, image_size]
        Returns:
            slots: [batch_size, num_slots, slot_size]
        """
        return self.encoder(x)
    
    def quantize(self, slots):
        """
        양자화 함수
        Args:
            slots: [batch_size, num_slots, slot_size]
        Returns:
            quantized_slots: [batch_size, num_slots, slot_size]
            commitment_loss: 커밋먼트 손실
            encoding_indices: [batch_size, num_slots, num_codebooks]
        """
        return self.quantizer(slots)
    
    def decode(self, slots):
        """
        디코딩 함수
        Args:
            slots: [batch_size, num_slots, slot_size]
        Returns:
            recon: [batch_size, in_channels, image_size, image_size]
            masks: [batch_size, num_slots, 1, image_size, image_size]
            slots_recon: [batch_size, num_slots, in_channels, image_size, image_size]
        """
        return self.decoder(slots)
    
    def forward(self, x):
        """
        전체 모델 순전파
        Args:
            x: [batch_size, in_channels, image_size, image_size]
        Returns:
            recon: [batch_size, in_channels, image_size, image_size]
            slots: [batch_size, num_slots, slot_size]
            quantized_slots: [batch_size, num_slots, slot_size]
            commitment_loss: 커밋먼트 손실
            encoding_indices: [batch_size, num_slots, num_codebooks]
            masks: [batch_size, num_slots, 1, image_size, image_size]
            slots_recon: [batch_size, num_slots, in_channels, image_size, image_size]
        """
        # 인코딩
        slots = self.encode(x)
        
        # 양자화
        quantized_slots, commitment_loss, encoding_indices = self.quantize(slots)
        
        # 디코딩
        recon, masks, slots_recon = self.decode(quantized_slots)
        
        return {
            'recon': recon,
            'slots': slots,
            'quantized_slots': quantized_slots,
            'commitment_loss': commitment_loss,
            'encoding_indices': encoding_indices,
            'masks': masks,
            'slots_recon': slots_recon
        }
    
    def compute_loss(self, x, outputs, recon_loss_weight=1.0, commitment_loss_weight=0.25):
        """
        손실 계산
        Args:
            x: [batch_size, in_channels, image_size, image_size]
            outputs: forward 함수의 출력
            recon_loss_weight: 재구성 손실 가중치
            commitment_loss_weight: 커밋먼트 손실 가중치
        Returns:
            total_loss: 총 손실
            loss_dict: 손실 구성요소 딕셔너리
        """
        recon = outputs['recon']
        commitment_loss = outputs['commitment_loss']
        
        # 재구성 손실 (MSE)
        recon_loss = F.mse_loss(recon, x)
        
        # 총 손실
        total_loss = recon_loss_weight * recon_loss + commitment_loss_weight * commitment_loss
        
        # 손실 구성요소 딕셔너리
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'commitment': commitment_loss.item()
        }
        
        return total_loss, loss_dict
    
    def generate(self, batch_size=1, temperature=1.0):
        """
        이미지 생성 (프라이어가 초기화된 경우에만 가능)
        Args:
            batch_size: 배치 크기
            temperature: 샘플링 온도
        Returns:
            generated_images: [batch_size, in_channels, image_size, image_size]
        """
        if self.prior is None:
            raise ValueError("Prior is not initialized. Call init_prior() first.")
        
        # 프라이어로부터 코드 샘플링
        encoding_indices = self.prior.generate(batch_size=batch_size, temperature=temperature)
        
        # 코드를 슬롯으로 변환
        codes = self.quantizer.get_codes_from_indices(encoding_indices)
        
        # 코드를 슬롯 차원으로 변환
        batch_size, num_slots, num_codebooks, code_dim = codes.shape
        
        # 각 코드북의 코드를 블록으로 변환
        blocks = []
        for i in range(num_codebooks):
            block_codes = codes[:, :, i]  # [batch_size, num_slots, code_dim]
            block = self.quantizer.output_projections[i](block_codes)  # [batch_size, num_slots, slot_size//num_codebooks]
            blocks.append(block)
        
        # 블록을 결합하여 슬롯 생성
        quantized_slots = torch.cat(blocks, dim=2)  # [batch_size, num_slots, slot_size]
        
        # 슬롯을 이미지로 디코딩
        generated_images, _, _ = self.decode(quantized_slots)
        
        return generated_images
    
    def get_codebook_usage_stats(self):
        """
        코드북 사용 통계 반환 (시각화 및 ablation용)
        """
        return self.quantizer.get_codebook_usage_stats()
    
    def reset_codebook_usage_stats(self):
        """
        코드북 사용 통계 초기화
        """
        self.quantizer.reset_usage_stats()
