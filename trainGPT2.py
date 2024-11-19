import os
import time
import math
import urllib.request
import tiktoken
import torch
import torch.nn as nn
from tqdm import tqdm

from Blocks.Configs import GPT_CONFIG_124M
from Models.GPT2 import GPT2


# Define a lightweight data loader class
class DataLoaderLite:
    def __init__(self, B, T, split="train"):
        """
        Lightweight data loader for tokenized text data, supporting train/validation splits.

        Args:
            B (int): Batch size.
            T (int): Sequence length.
            split (str): 'train' or 'val' to specify data split.
        """
        self.B = B
        self.T = T
        self.split = split

        # File path and URL for the dataset
        file_path = "Hamlet.txt"
        url = "https://raw.githubusercontent.com/Mu7annad0/LLMForge/refs/heads/main/Hamlet.txt"
        
        # Download or load the dataset
        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

        # Tokenize the text using GPT-2's encoding
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
        
        # Initialize the position tracker for the next batch
        self.current_position = 0
        
        print(f"Total number of tokens for {split}: {len(self.tokens)}")
        self.n_batches = len(self.tokens) // (self.B * self.T)
        print(f"Number of batches per epoch for {split}: {self.n_batches}")

    def next_batch(self):
        """
        Retrieves the next batch of data for training/validation.

        Returns:
            x (torch.Tensor): Input tensor of shape (B, T).
            y (torch.Tensor): Target tensor of shape (B, T), shifted by one token.
        """
        B, T = self.B, self.T

        # Extract a buffer of tokens and create input (x) and target (y) tensors
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # Input tokens
        y = (buf[1:]).view(B, T)  # Target tokens, shifted one position

        # Update position for the next batch, resetting if at the end of data
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
        
    def __len__(self):
        return self.n_batches


# Utility functions
def get_device():
    """
    Determines the available device for PyTorch computations.

    Returns:
    str: The device string, either 'cuda', 'mps', or 'cpu'.
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
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


# Set the learning rate of each parameter group using a cosine annealing schedule.
def get_cosine_annealed_lr(iter):
    """
    Computes the learning rate using a cosine annealing schedule with an initial warm-up phase.

    Args:
        iter (int): Current training iteration (step).

    Returns:
        float: Computed learning rate for the given iteration.
    """
    # If we're still in the warm-up phase, gradually increase the learning rate from 0 to max_lr.
    if iter < warmup_steps:
        return max_lr * (iter + 1) / warmup_steps
    # Calculate the ratio of progress after the warmup phase
    # decay_ratio ranges from 0 to 1 as iter goes from warmup_steps to max_steps
    decay_ratio = min(1.0, (iter - warmup_steps) / (max_steps - warmup_steps))
    # Return the learning rate following a cosine decay function
    # The cosine function smoothly decays the learning rate from max_lr to min_lr
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay_ratio))


# Training function
def train_gpt(Model):
    # Get the device (CPU, CUDA, or MPS) and move the model to that device
    device = get_device()
    Model.to(device)
    # Initialize the optimizer with AdamW
    optimizer = Model.configure_optimizers(weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95), device_type=device)
    # Calculate gradient accumulation steps based on total batch size
    grad_accum_steps = total_batch_size // (batch_size * max_seq_length)
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Set up the logging directory and file
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass # Create or clear the log file

    # Training loop
    for step in tqdm(range(max_steps), desc='Training'):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Evaluate model on validation set every 100 steps or at the last step
        if step % 100 == 0 or last_step:
                Model.eval() # Set model to evaluation mode
                with torch.no_grad():
                    val_loss_accumulation = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = valid_loader.next_batch()
                        x, y = x.to(device), y.to(device)

                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            logits, loss = Model(x, y)
                        loss /= val_loss_steps # Normalize loss by the number of steps
                        val_loss_accumulation += loss.detach()

                # Save model checkpoint every 5000 steps or at the last step
                if step > 0 and (step % 5000 == 0 or last_step):
                    ck_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': Model.state_dict(),
                        'config': GPT_CONFIG_124M,
                        'step': step,
                        'val_loss': val_loss_accumulation.item()
                    }
                    torch.save(checkpoint, ck_path)

                # Log validation loss
                print(f"-->> Validation loss: {val_loss_accumulation.item():.5f}")
                with open(log_file, "a") as f:
                    f.write(f"validation loss is : {val_loss_accumulation.item():.5f}\n")
            
        # Training logic
        Model.train() # Set model to training mode
        optimizer.zero_grad()
        loss_accumulation = 0.0

        # Progress bar for micro-batches
        with tqdm(total=grad_accum_steps, desc=f'Step {step + 1}/{max_steps}', leave=False) as micro_progress:
            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)

                # Forward pass with mixed precision
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = Model(x, y)
                
                loss /= grad_accum_steps 
                loss_accumulation += loss.detach()
                loss.backward()

                # Update progress bar with loss and learning rate
                micro_progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{get_cosine_annealed_lr(step):.6f}'
                })
                micro_progress.update(1)

        # Gradient clipping to track the stability
        norm = nn.utils.clip_grad_norm_(Model.parameters(), 1.0)
        # Applying the cosine annealing learning rate scheduler
        lr = get_cosine_annealed_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr # Update learning rate in optimizer

        optimizer.step() # Apply gradient update

        # Synchronize device for timing if CUDA or MPS is used
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()

        dt = (time.time() - t0) * 1000
        tokens_per_sec = (batch_size * max_seq_length * grad_accum_steps) / (dt / 1000)

        # Log training progress
        print(f"Step {step + 1}/{max_steps} - loss: {loss_accumulation.item():.5f}, norm: {norm:.4f}, "
            f"lr: {lr:.6f}, dt: {dt:.2f}ms, tokens/sec: {tokens_per_sec:.0f}")
        with open(log_file, "a") as f:
            f.write(f"--> {step}/{max_steps} train loss is : {loss_accumulation.item():.5f}\n")


# Main script
if __name__ == "__main__":
    set_seed()

    batch_size, max_seq_length = 8, 128
    total_batch_size = 524288 # total batch size used for gradient accumulation

    train_loader = DataLoaderLite(batch_size, max_seq_length, "train")
    valid_loader = DataLoaderLite(batch_size, max_seq_length, "val")

    torch.set_float32_matmul_precision("high")

    # Instantiate the GPT-2 model
    cfg = GPT_CONFIG_124M(vocab_size=50304)
    Model = GPT2(cfg)

    max_steps = 200         # Total number of training steps
    warmup_steps = 10       # Number of steps for learning rate warm-up
    max_lr = 6e-4           # Maximum learning rate
    min_lr = max_lr * 0.1   # Minimum learning rate (10% of max_lr for cosine annealing)

    # Start the training process
    train_gpt(Model=Model)
