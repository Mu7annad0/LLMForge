import torch
import torch.nn as nn

from Attentions.CausalAttention import CausalAttention
from Attentions.GroupedQueryAttention import GrupedQueryAttention
from .FeedForwards import GPTFeedForward, LlamaFeedForward
from .Activations import GELU
from .Normalizations import LayerNormalization, RMSNormalization


class GPTTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = CausalAttention(
            embed_dim=cfg.embed_dim,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            num_heads=cfg.n_heads,
            qkv=cfg.qkv_bias
        )
        self.ffn = GPTFeedForward(cfg)
        self.lnorm = LayerNormalization(cfg.embed_dim)
        self.gelu = nn.GELU(approximate="tanh")
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
    

class LlamaTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.rms = RMSNormalization(cfg.embed_dim)
        self.attn = GrupedQueryAttention(cfg)
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