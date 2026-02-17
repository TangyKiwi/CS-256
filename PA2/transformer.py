from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    def __init__(self, embed_size: int, head_dim: int):
        super().__init__()
        self.key = nn.Linear(embed_size, head_dim, bias=False)
        self.query = nn.Linear(embed_size, head_dim, bias=False)
        self.value = nn.Linear(embed_size, head_dim, bias=False)
        self.head_dim = head_dim
    
    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        K = self.key(x)  # (B, T, head_dim)
        Q = self.query(x)  # (B, T, head_dim)
        V = self.value(x)  # (B, T, head_dim)

        # Compute attention scores
        att = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, T, T)

        allowed = torch.ones((B, T, T), device=x.device, dtype=torch.bool)

        if causal:
            causal_mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool))
            allowed &= causal_mask.unsqueeze(0)

        if mask is not None:
            allowed &= mask[:, None, :]

        att = att.masked_fill(~allowed, float('-inf'))  # Mask out disallowed positions
        att = F.softmax(att, dim=-1)  # (B, T, T)
        out = att @ V  # (B, T, head_dim)

        return out, att
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        head_dim = embed_size // num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(embed_size, head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_outputs = []
        att_maps = []
        for head in self.heads:
            out, att_map = head(x, mask, causal=causal)
            head_outputs.append(out)
            att_maps.append(att_map)

        out = torch.cat(head_outputs, dim=-1)  # (B, T, embed_size)
        out = self.proj(out)  # (B, T, embed_size)

        return out, att_maps
    
class FeedForward(nn.Module):
    def __init__(self, embed_size: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.mhsa = MultiHeadSelfAttention(embed_size, num_heads)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ffn = FeedForward(embed_size, hidden_dim)
        
    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        att_out, att_maps = self.mhsa(self.ln1(x), mask)
        x = x + att_out
        x = x + self.ffn(self.ln2(x))
        return x, att_maps
    
class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            block_size: int,
            embed_size: int, 
            num_heads: int, 
            num_layers: int,
            hidden_dim: Optional[int] = None, 
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_size = embed_size

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size, embed_size)

        if hidden_dim is None:
            hidden_dim = 4 * embed_size

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_size, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        

    def forward(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T = x.shape
        if T > self.block_size:
            x = x[:, :self.block_size]
            T = self.block_size

        mask = (x != 0) # true where token is real, false where PAD = 0
        pos = torch.arange(0, T, device=x.device, dtype=torch.long) # (T,)
        x = self.token_embedding(x) + self.pos_embedding(pos)[None, :, :]  # (B, T, embed_size)

        att_maps_all = []
        for block in self.blocks:
            x, att_maps = block(x, mask)
            att_maps_all.extend(att_maps)
        
        x = self.ln_f(x)  # (B, T, embed_size)
        return x, att_maps_all
    
class FeedForwardClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, _ = self.encoder(x)  # (B, T, embed_size)
        pooled = enc_out.mean(dim=1)  # (B, embed_size)
        
        x = self.fc1(pooled)  # (B, hidden_dim)
        x = F.relu(x)
        return self.fc2(x)  # (B, output_dim)
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, hidden_dim: int = 100):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.mhsa = MultiHeadSelfAttention(embed_size, num_heads)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ffn = FeedForward(embed_size, hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        att_out, att_maps = self.mhsa(self.ln1(x), mask, causal=True)
        x = x + att_out
        x = x + self.ffn(self.ln2(x))
        return x, att_maps
    
class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            block_size: int,
            embed_size: int, 
            num_heads: int, 
            num_layers: int,
            hidden_dim: Optional[int] = 100, 
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_size = embed_size

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size, embed_size)

        if hidden_dim is None:
            hidden_dim = 4 * embed_size

        self.blocks = nn.ModuleList([
            DecoderBlock(embed_size, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(
            self,
            x: torch.Tensor,
            targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T = x.shape
        if T > self.block_size:
            x = x[:, :self.block_size]
            T = self.block_size
            if targets is not None:
                targets = targets[:, :self.block_size]

        mask = (x != 0) # true where token is real, false where PAD = 0
        pos = torch.arange(0, T, device=x.device, dtype=torch.long) # (T,)
        x = self.token_embedding(x) + self.pos_embedding(pos)[None, :, :]  # (B, T, embed_size)

        att_maps_all = []
        for block in self.blocks:
            x, att_maps = block(x, mask)
            att_maps_all.extend(att_maps)
        
        x = self.ln_f(x)  # (B, T, embed_size)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, att_maps_all
        
        loss = F.cross_entropy(
            logits.view(B * T, self.vocab_size),
            targets.view(B * T),
        )

        return loss