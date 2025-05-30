import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SlotAttention(nn.Module):
    """
    슬롯 어텐션 모듈 구현
    논문: "Slot Attention" (Locatello et al., 2020)
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        """
        Args:
            num_slots: 슬롯 수
            dim: 슬롯 차원
            iters: 반복 횟수
            eps: 수치 안정성을 위한 작은 값
            hidden_dim: MLP 은닉층 차원
        """
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.gru = nn.GRUCell(dim, dim)
        
        hidden_dim = max(dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        """
        Args:
            inputs: [batch_size, num_inputs, dim]
            num_slots: 슬롯 수 (None인 경우 self.num_slots 사용)
        Returns:
            slots: [batch_size, num_slots, dim]
        """
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        
        # 슬롯 초기화
        slots = torch.randn(b, n_s, d, device=inputs.device)
        slots = self.slots_mu + self.slots_sigma * slots
        
        for _ in range(self.iters):
            slots_prev = slots
            
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            
            # 어텐션
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            updates = torch.einsum('bij,bjd->bid', attn, v)
            
            # GRU 업데이트
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            
        return slots


class SlotAttentionEncoder(nn.Module):
    """
    슬롯 어텐션 인코더
    """
    def __init__(
        self,
        image_size=64,
        num_slots=4,
        num_iterations=3,
        in_channels=3,
        slot_size=64,
        hidden_dim=64,
        pos_embed_type='learned'
    ):
        """
        Args:
            image_size: 입력 이미지 크기
            num_slots: 슬롯 수
            num_iterations: 슬롯 어텐션 반복 횟수
            in_channels: 입력 채널 수
            slot_size: 슬롯 차원
            hidden_dim: CNN 은닉층 차원
            pos_embed_type: 위치 임베딩 타입 ('learned' 또는 'fixed')
        """
        super().__init__()
        self.image_size = image_size
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.slot_size = slot_size
        
        # CNN 인코더 - 출력 크기를 명확히 하기 위해 stride 추가
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2),  # 출력 크기: image_size/2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),   # 출력 크기: image_size/2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),   # 출력 크기: image_size/4
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, slot_size, kernel_size=5, stride=1, padding=2),    # 출력 크기: image_size/4
        )
        
        self.pos_embed_type = pos_embed_type
        self.pos_embed = None
        
        # 동적으로 위치 임베딩 크기 계산
        self.feature_size = image_size // 4  # CNN 인코더 출력 크기
        
        if pos_embed_type == 'learned':
            self.pos_embed = nn.Parameter(torch.randn(1, slot_size, self.feature_size, self.feature_size))
        elif pos_embed_type == 'fixed':
            self.pos_embed = self._build_fixed_pos_embed(slot_size, self.feature_size)
        
        self.layer_norm = nn.LayerNorm(slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_size)
        )
        
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=slot_size,
            iters=num_iterations,
            hidden_dim=hidden_dim
        )
    
    def _build_fixed_pos_embed(self, dim, size):
        """
        고정 위치 임베딩 생성
        """
        pos_embed = torch.zeros(1, dim, size, size)
        for i in range(size):
            for j in range(size):
                for d in range(dim):
                    if d % 2 == 0:
                        pos_embed[0, d, i, j] = torch.sin(torch.tensor(i / 10000 ** (d / dim)))
                    else:
                        pos_embed[0, d, i, j] = torch.cos(torch.tensor(i / 10000 ** ((d-1) / dim)))
        return nn.Parameter(pos_embed, requires_grad=False)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, in_channels, image_size, image_size]
        Returns:
            slots: [batch_size, num_slots, slot_size]
        """
        batch_size = x.shape[0]
        
        # CNN 인코딩
        x = self.encoder_cnn(x)  # [batch_size, slot_size, feature_size, feature_size]
        
        # 디버깅을 위한 shape 출력
        print(f"CNN 출력 shape: {x.shape}, pos_embed shape: {self.pos_embed.shape}")
        
        # 위치 임베딩 추가 - 크기가 맞지 않을 경우 동적으로 조정
        if self.pos_embed is not None:
            if x.shape[2:] != self.pos_embed.shape[2:]:
                pos_embed = F.interpolate(
                    self.pos_embed, 
                    size=x.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                x = x + pos_embed
            else:
                x = x + self.pos_embed
        
        # 슬롯 어텐션 입력 형태로 변환
        x = rearrange(x, 'b d h w -> b (h w) d')  # [batch_size, feature_size*feature_size, slot_size]
        
        # 정규화 및 MLP
        x = self.layer_norm(x)
        x = self.mlp(x)
        
        # 슬롯 어텐션 적용
        slots = self.slot_attention(x, self.num_slots)  # [batch_size, num_slots, slot_size]
        
        return slots
