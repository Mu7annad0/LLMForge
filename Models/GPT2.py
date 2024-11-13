import inspect
import torch
import torch.nn as nn

from Blocks.Transformers import GPTTransformer
from Blocks.Normalizations import LayerNormalization

class GPT2(nn.Module):
    def __init__(self, cfg):
        """
        Initializes the GPT-2 model with the specified configuration.

        Args:
        cfg: A configuration object that includes:
            - vocab_size (int): The size of the vocabulary.
            - embed_dim (int): The dimensionality of the embeddings.
            - context_length (int): The maximum length of input sequences.
            - drop_rate (float): Dropout rate to use in the model.
            - n_layers (int): Number of transformer layers.
        """
        super().__init__()

        # Token and positional embedding layers
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.embed_dim)
        # Dropout layer for regularization
        self.drop = nn.Dropout(cfg.drop_rate)
        # Stack of transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[GPTTransformer(cfg) for _ in range(cfg.n_layers)]
        )
        # Layer normalization before final output
        self.lnorm = LayerNormalization(cfg.embed_dim)
        # Output projection layer to map to vocabulary size
        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_size)
        # Weight sharing scheme
        self.token_embed.weight = self.output_layer.weight

    def forward(self, x, targets=None):
        """
        Forward pass of the GPT-2 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): Target tensor for computing loss, 
                                           of shape (batch_size, sequence_length).

        Returns:
            logits (torch.Tensor): Output logits of shape (batch_size, sequence_length, vocab_size).
            loss: Cross-entropy loss if targets are provided; otherwise, None.
        """
        B, seq_len = x.shape
        # Create token and positional embeddings
        token_embed = self.token_embed(x)
        pos_embed = self.pos_embed(torch.arange(seq_len, device=x.device))
        # Combine token and positional embeddings and apply dropout
        x = token_embed + pos_embed
        x = self.drop(x)
        # Pass through the stack of transformer blocks
        x = self.transformer_blocks(x)
        # Apply final layer normalization
        x = self.lnorm(x)
        logits = self.output_layer(x)
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), # Reshape logits for cross-entropy
                targets.view(-1) # Reshape targets to match logits
            )
        return logits, loss
    
    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        """
        Configures the optimizer for training the model.

        Args:
            weight_decay (float): Weight decay (L2 regularization) to apply to certain parameters.
            lr (float): Learning rate for the optimizer.
            betas (tuple): Coefficients used for computing running averages of gradient and its square.
            device_type (str): The type of device being used ('cpu', 'cuda', or 'mps').

        Returns:
            optimizer (torch.optim.Optimizer): The configured AdamW optimizer.
        """
        # Collect all trainable parameters as a dictionary with their names
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}

        # Separate parameters into two groups:
        # - decay_params: Parameters that will receive weight decay (e.g., weights of linear layers)
        # - nodecay_params: Parameters that will not receive weight decay (e.g., biases and layer norms)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # Create optimizer parameter groups with specific weight decay settings
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Log the number of parameters in each group for debugging and transparency
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameter tensors is: {len(decay_params)}, with {num_decay_params:,} parmaters")
        print(f"Num nun-decayed parameter tensors is: {len(nodecay_params)}, with {num_nodecay_params:,} parmeters")

        # Check if the 'fused' argument is supported by AdamW (useful for speed optimization)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        
        # Use fused AdamW if available and running on a GPU-like device
        use_fused = fused_available and device_type == ('mps' or 'cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        print(f"using fused AdamW: {use_fused}")

        # Create the AdamW optimizer with the specified parameter groups and settings
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)

        return optimizer