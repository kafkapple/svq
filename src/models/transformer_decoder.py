import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TransformerDecoder(nn.Module):
    """
    트랜스포머 디코더 모듈
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        ff_dim=None
    ):
        """
        Args:
            dim: 입력 및 출력 차원
            num_heads: 어텐션 헤드 수
            num_layers: 트랜스포머 레이어 수
            dropout: 드롭아웃 비율
            ff_dim: 피드포워드 네트워크 은닉층 차원 (None인 경우 dim*4 사용)
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        if ff_dim is None:
            ff_dim = dim * 4
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, memory=None, mask=None):
        """
        Args:
            x: [batch_size, seq_len, dim]
            memory: [batch_size, mem_len, dim] (None인 경우 self-attention만 수행)
            mask: [batch_size, seq_len, seq_len] (None인 경우 마스크 없음)
        Returns:
            output: [batch_size, seq_len, dim]
        """
        for layer in self.layers:
            x = layer(x, memory, mask)
        
        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    """
    트랜스포머 디코더 레이어
    """
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        """
        Args:
            dim: 입력 및 출력 차원
            num_heads: 어텐션 헤드 수
            ff_dim: 피드포워드 네트워크 은닉층 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention (memory가 제공된 경우)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, memory=None, mask=None):
        """
        Args:
            x: [batch_size, seq_len, dim]
            memory: [batch_size, mem_len, dim] (None인 경우 cross-attention 건너뜀)
            mask: [batch_size, seq_len, seq_len] (None인 경우 마스크 없음)
        Returns:
            output: [batch_size, seq_len, dim]
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout1(x)
        
        # Cross-attention (memory가 제공된 경우)
        if memory is not None:
            residual = x
            x = self.norm2(x)
            x, _ = self.cross_attn(x, memory, memory)
            x = residual + self.dropout2(x)
        
        # Feed-forward network
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + self.dropout3(x)
        
        return x


class AutoregressiveTransformer(nn.Module):
    """
    오토리그레시브 트랜스포머 모델
    SVQ 논문의 시맨틱 프라이어(Semantic Prior) 구현
    """
    def __init__(
        self,
        num_slots,
        num_codebooks,
        codebook_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ):
        """
        Args:
            num_slots: 슬롯 수 (N)
            num_codebooks: 코드북 수 (M)
            codebook_size: 각 코드북의 크기 (K)
            embed_dim: 임베딩 차원
            num_heads: 어텐션 헤드 수
            num_layers: 트랜스포머 레이어 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        
        # 시작 토큰 임베딩
        self.bos_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 코드 임베딩
        self.code_embed = nn.Embedding(codebook_size, embed_dim)
        
        # 위치 임베딩
        self.pos_embed = nn.Parameter(torch.randn(1, num_slots * num_codebooks + 1, embed_dim))
        
        # 슬롯 인덱스 임베딩
        self.slot_embed = nn.Embedding(num_slots, embed_dim)
        
        # 코드북 인덱스 임베딩
        self.codebook_embed = nn.Embedding(num_codebooks, embed_dim)
        
        # 트랜스포머 디코더
        self.transformer = TransformerDecoder(
            dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 출력 헤드
        self.output_head = nn.Linear(embed_dim, codebook_size)
    
    def _create_causal_mask(self, seq_len, device):
        """
        인과적 마스크 생성 (미래 토큰을 볼 수 없도록)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, codes=None, generate=False, temperature=1.0):
        """
        Args:
            codes: [batch_size, num_slots, num_codebooks] (None인 경우 생성 모드)
            generate: 생성 모드 여부
            temperature: 샘플링 온도
        Returns:
            logits: [batch_size, seq_len, codebook_size] (학습 모드)
            generated_codes: [batch_size, num_slots, num_codebooks] (생성 모드)
        """
        device = self.bos_token.device
        
        if generate:
            return self.generate(temperature=temperature)
        
        batch_size = codes.shape[0]
        
        # 코드를 시퀀스로 변환 [batch_size, num_slots, num_codebooks] -> [batch_size, num_slots*num_codebooks]
        codes_flat = rearrange(codes, 'b n m -> b (n m)')
        
        # BOS 토큰 추가
        input_seq = torch.cat([
            torch.zeros(batch_size, 1, device=device, dtype=torch.long),
            codes_flat
        ], dim=1)
        
        # 입력 임베딩
        token_embeds = torch.cat([
            self.bos_token.expand(batch_size, 1, -1),
            self.code_embed(codes_flat)
        ], dim=1)
        
        # 위치 임베딩 추가
        token_embeds = token_embeds + self.pos_embed
        
        # 슬롯 및 코드북 인덱스 임베딩 추가
        slot_indices = torch.arange(self.num_slots, device=device).expand(batch_size, self.num_slots)
        codebook_indices = torch.arange(self.num_codebooks, device=device).expand(batch_size, self.num_codebooks)
        
        slot_embeds = self.slot_embed(slot_indices)  # [batch_size, num_slots, embed_dim]
        codebook_embeds = self.codebook_embed(codebook_indices)  # [batch_size, num_codebooks, embed_dim]
        
        # 슬롯 및 코드북 임베딩 조합
        slot_codebook_embeds = torch.zeros(batch_size, self.num_slots * self.num_codebooks, self.embed_dim, device=device)
        
        for i in range(self.num_slots):
            for j in range(self.num_codebooks):
                idx = i * self.num_codebooks + j
                slot_codebook_embeds[:, idx] = slot_embeds[:, i] + codebook_embeds[:, j]
        
        # BOS 토큰 임베딩과 결합
        combined_embeds = torch.cat([
            self.bos_token.expand(batch_size, 1, -1),
            slot_codebook_embeds
        ], dim=1)
        
        # 최종 입력 임베딩
        input_embeds = token_embeds + combined_embeds
        
        # 인과적 마스크 생성
        seq_len = input_embeds.shape[1]
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # 트랜스포머 디코더 적용
        output = self.transformer(input_embeds, mask=causal_mask)
        
        # 출력 헤드 적용
        logits = self.output_head(output)
        
        # BOS 토큰 제외
        logits = logits[:, :-1]
        
        return logits
    
    def generate(self, batch_size=1, temperature=1.0):
        """
        오토리그레시브 방식으로 코드 생성
        Args:
            batch_size: 배치 크기
            temperature: 샘플링 온도
        Returns:
            generated_codes: [batch_size, num_slots, num_codebooks]
        """
        device = self.bos_token.device
        
        # 생성된 코드를 저장할 텐서
        generated_codes = torch.zeros(batch_size, self.num_slots * self.num_codebooks, dtype=torch.long, device=device)
        
        # 시작 토큰으로 초기화
        current_input = self.bos_token.expand(batch_size, 1, -1)
        
        # 위치 임베딩 준비
        pos_embed = self.pos_embed[:, :1]
        
        # 슬롯 및 코드북 인덱스 임베딩 준비
        slot_indices = torch.arange(self.num_slots, device=device).expand(batch_size, self.num_slots)
        codebook_indices = torch.arange(self.num_codebooks, device=device).expand(batch_size, self.num_codebooks)
        
        slot_embeds = self.slot_embed(slot_indices)  # [batch_size, num_slots, embed_dim]
        codebook_embeds = self.codebook_embed(codebook_indices)  # [batch_size, num_codebooks, embed_dim]
        
        # 오토리그레시브 생성
        for i in range(self.num_slots * self.num_codebooks):
            # 현재 슬롯 및 코드북 인덱스 계산
            slot_idx = i // self.num_codebooks
            codebook_idx = i % self.num_codebooks
            
            # 슬롯 및 코드북 임베딩 조합
            combined_embed = slot_embeds[:, slot_idx] + codebook_embeds[:, codebook_idx]
            combined_embed = combined_embed.unsqueeze(1)  # [batch_size, 1, embed_dim]
            
            # 현재 입력에 위치 임베딩과 슬롯/코드북 임베딩 추가
            current_pos_embed = self.pos_embed[:, :current_input.shape[1]]
            input_embeds = current_input + current_pos_embed
            
            # 마지막 토큰에 대해서만 슬롯/코드북 임베딩 추가
            input_embeds[:, -1:] = input_embeds[:, -1:] + combined_embed
            
            # 인과적 마스크 생성
            seq_len = input_embeds.shape[1]
            causal_mask = self._create_causal_mask(seq_len, device)
            
            # 트랜스포머 디코더 적용
            output = self.transformer(input_embeds, mask=causal_mask)
            
            # 마지막 토큰에 대한 예측
            logits = self.output_head(output[:, -1])
            
            # 온도 적용 및 샘플링
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            # 생성된 코드 저장
            generated_codes[:, i] = next_token
            
            # 다음 입력 준비
            next_token_embed = self.code_embed(next_token).unsqueeze(1)
            current_input = torch.cat([current_input, next_token_embed], dim=1)
        
        # 생성된 코드를 원래 형태로 변환 [batch_size, num_slots*num_codebooks] -> [batch_size, num_slots, num_codebooks]
        generated_codes = generated_codes.view(batch_size, self.num_slots, self.num_codebooks)
        
        return generated_codes
