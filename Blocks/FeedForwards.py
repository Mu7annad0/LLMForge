import torch 
import torch.nn as nn

from .Activations import GELU, SiLU


class GPTFeedForward(nn.Module):

    def __init__(self, cfg):
        """
        The GPTFeedForward block is a component of the transformer architecture. 
        It consists of two linear layers with a GELU activation function in between. 
        The first linear layer expands the input dimension by a factor of 4, followed by the GELU activation. 
        The second linear layer reduces the dimension back to the original size. 
        This structure allows the model to learn complex representations of the input data. 
        """
        super().__init__()
        self.embed_dim = cfg.embed_dim  
        # Define the sequence of layers for the feed forward block
        self.layers = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),  # First linear layer expands the dimension by a factor of 4
            GELU(),  # GELU activation function for non-linearity
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim)  # Second linear layer reduces the dimension back to the original size
        )

    def forward(self, x):
        return self.layers(x)  # Forward pass through the defined layers


class LlamaFeedForward(nn.Module):
    
    def __init__(self, cfg):
        """
        The LlamaFeedForward block is a transformer component with three linear layers and a SiLU activation. 
        The first two layers expand to the hidden dimension, with SiLU in between. 
        The third layer reduces the dimension back to the original size after an element-wise product of the first two layers' outputs. 
        This structure enables complex input data representations.
        """
        super().__init__()

        self.layer1 = nn.Linear(cfg.embed_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False) # dtype: allowing the model to be loaded in lower precision format to save memory
        self.layer2 = nn.Linear(cfg.embed_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False) # Llama doesn't use any bias
        self.layer3 = nn.Linear(cfg.hidden_dim, cfg.embed_dim, dtype=cfg.dtype, bias=False)
        self.silu = SiLU()

    def forward(self, x):
        x1 = self.layer1(x)
        x = self.silu(x1) * self.layer2(x)
        return self.layer3(x)