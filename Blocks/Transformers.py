import torch
import torch.nn as nn

from Attentions.MultiHeadAttention import MultiHeadAttention
from Attentions.AttentionRoPE import MultiHeadAttentionRoPE
from .FeedForwards import GPTFeedForward, LlamaFeedForward
from .Activations import GELU
from .Normalizations import LayerNormalization, RMSNormalization


class GPTTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_in=cfg.embed_dim,
            d_out=cfg.embed_dim,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            num_heads=cfg.n_heads,
            qkv=cfg.qkv_bias
        )
        self.ffn = GPTFeedForward(cfg)
        self.lnorm = LayerNormalization(cfg.embed_dim)
        self.gelu = GELU()
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.lnorm(x)
        x = self.attn(x)
        x = self.drop(x)
        x += shortcut

        shortcut = x
        x = self.lnorm(x)
        x = self.ffn(x)
        x = self.drop(x)
        x += shortcut

        return x
    

class Llama2Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.rms = RMSNormalization(cfg.embed_dim)
        self.attn = MultiHeadAttentionRoPE(
            d_in = cfg.embed_dim,
            d_out = cfg.embed_dim,
            context_length= cfg.context_length,
            num_heads= cfg.n_heads,
            dtype= cfg.dtype
        )
        self.ffn = LlamaFeedForward(cfg)

    def forward(self, x):
        shortcut = x
        x = self.rms(x)
        x = self.attn(x)
        x += shortcut

        shortcut = x
        x = self.rms(x)
        x = self.ffn(x)
        x += shortcut

        return x