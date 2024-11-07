import os
import time
import math
import urllib.request
import tiktoken
import torch
import torch.nn as nn
from tqdm import tqdm

from Blocks.Transformers import GPTTransformer
from Blocks.Normalizations import LayerNormalization
from Blocks.Configs import GPT_CONFIG_124M


class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[GPTTransformer(cfg) for _ in range(cfg.n_layers)]
        )
        self.lnorm = LayerNormalization(cfg.embed_dim)
        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_size)
        # weight sharing scheme
        self.token_embed.weight = self.output_layer.weight

    def forward(self, x, targets=None):
        B, seq_len = x.shape
        token_embed = self.token_embed(x)
        pos_embed = self.pos_embed(torch.arange(seq_len, device=x.device))
        x = token_embed + pos_embed
        x = self.drop(x)
        x = self.transformer_blocks(x)
        x = self.lnorm(x)
        logits = self.output_layer(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        return logits, loss


class DataLoaderLite:
    def __init__(self, B, T, split="train"):
        self.B = B
        self.T = T
        self.split = split

        file_path = "Hamlet.txt"
        url = "https://raw.githubusercontent.com/Mu7annad0/LLMForge/refs/heads/main/Hamlet.txt"
        
        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        all_tokens = torch.tensor(tokens)
        
        # Split into train (80%) and validation (20%) 
        split_idx = int(0.8 * len(all_tokens))
        
        if split == "train":
            self.tokens = all_tokens[:split_idx]
        elif split == "val":
            self.tokens = all_tokens[split_idx:]
        else:
            raise ValueError("Split must be either 'train' or 'val'")
            
        self.current_position = 0
        
        print(f"Total number of tokens for {split}: {len(self.tokens)}")
        self.n_batches = len(self.tokens) // (self.B * self.T)
        print(f"Number of batches per epoch for {split}: {self.n_batches}")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
        
    def __len__(self):
        return self.n_batches


def get_device():
    # Default to CPU
    device = "cpu"

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
    # Check for MPS availability (Apple Silicon)
    elif torch.backends.mps.is_available():
        device = "mps"

    print(f"Using device: {device}")
    return device


def set_seed(seed=2049):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed()

batch_size, max_seq_length = 8, 128
total_batch_size = 524288

train_loader = DataLoaderLite(batch_size, max_seq_length)
valid_loader = DataLoaderLite(batch_size, max_seq_length, "val")

torch.set_float32_matmul_precision("high")

Model = GPT2(GPT_CONFIG_124M(vocab_size=50304))

warmup_steps = 10
max_steps = 50
max_lr = 6e-4
min_lr = max_lr * 0.1


def get_lr(iter):
    # Linear warmup phase
    if iter < warmup_steps:
        return max_lr * (iter + 1) / warmup_steps
    
    # Cosine decay phase
    decay_ratio = min(1.0, (iter - warmup_steps) / (max_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay_ratio))


grad_accum_steps = 1

def train_gpt():
    device = get_device()
    Model.to(device)
    optimizer = torch.optim.AdamW(Model.parameters(), weight_decay=0.1, lr=3e-4, betas=(0.9, 0.95))
    grad_accum_steps = total_batch_size // (batch_size * max_seq_length)
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    for step in tqdm(range(max_steps), desc='Training'):
        t0 = time.time()
        
        # Reset gradients and loss accumulation for each step
        optimizer.zero_grad()
        loss_accumulation = 0.0

        with tqdm(total=grad_accum_steps, desc=f'Step {step + 1}/{max_steps}', leave=False) as micro_progress:
        # Perform forward and backward passes over `grad_accum_steps` micro-batches
            for _ in range(grad_accum_steps):
                # Get batch and move to device
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = Model(x, y)
                
                loss /= grad_accum_steps  # Scale the loss
                loss_accumulation += loss.detach()  # Accumulate loss for logging
                loss.backward()  # Accumulate gradients
                # Update the micro-step progress bar
                micro_progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{get_lr(step):.6f}'
                })
                micro_progress.update(1)
        # Apply gradient clipping and optimization
        norm = nn.utils.clip_grad_norm_(Model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # Synchronize device for accurate timing
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()

        # Calculate elapsed time and tokens per second
        dt = (time.time() - t0) * 1000  # Time in milliseconds
        tokens_per_sec = (batch_size * max_seq_length * grad_accum_steps) / (dt / 1000)

        # Print data for each step in the desired format
        print(f"Step {step + 1}/{max_steps} - loss: {loss_accumulation.item():.5f}, norm: {norm:.4f}, "
            f"lr: {lr:.6f}, dt: {dt:.2f}ms, tokens/sec: {tokens_per_sec:.0f}")


if __name__ == "__main__":
    train_gpt()