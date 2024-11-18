import torch
import torch.nn as nn
from dataclasses import dataclass

from Blocks.Normalizations import RMSNormalization
from Blocks.Positionals import precompute_rope_params, apply_rope

class DifferntialAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.init_lambda = cfg.lambda_init
        self.context_length = cfg.context_length
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads

        self.w_qkv = nn.Linear(cfg.embed_dim, 3*cfg.embed_dim, bias=False)
        self.wo = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)

        self.norm = nn.ModuleList(
            RMSNormalization(self.head_dim) for _ in range(self.num_heads)
        )

        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim))

    def forward(self, x):
        bsz, seqlen, embed_dim = x.shape
        qkv = self.w_qkv(x).chunk(3, dim=-1)
        q, k, v = [x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2) for x in qkv]

        q1, q2 = torch.chunk(input=q, chunks=2, dim=-1)
        k1, k2 = torch.chunk(input=k, chunks=2, dim=-1)

        a1, a2 = q1 @ k1.transpose(2, 3), q2 @ k2.transpose(2, 3)
        a1, a2 = torch.softmax((a1/self.head_dim ** 0.5), dim=-1), torch.softmax((a2/self.head_dim ** 0.5), dim=-1)

        lambda_1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
        lambda_2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
        lambda_ = lambda_1 - lambda_2 + self.init_lambda

        a = a1 - (lambda_ * a2)
        out = a @ v

        self.norms_heads = []
        for i, norm in enumerate(self.norm):
            self.norms_heads.append(norm(out[:,i]) * (1 - self.init_lambda))
        
        out = torch.cat(self.norms_heads, dim=1)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, embed_dim)
        out = self.wo(out)

        return out
        

@dataclass
class config:
    batch_size = 1
    context_length = 512
    embed_dim = 256
    num_heads = 4
    lambda_init = 0.8

cfg = config
mha = DifferntialAttention(cfg)

example_batch = torch.randn((cfg.batch_size, cfg.context_length, cfg.embed_dim))
out = mha(example_batch)

print(f"{out.shape=}")