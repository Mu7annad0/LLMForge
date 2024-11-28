import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

from Blocks.Transformers import LlamaTransformer
from Blocks.Normalizations import RMSNormalization
from Blocks.Configs import Llama31_CONFIG_8B
from tokenizer import Tokenizer


class Llama(nn.Module):
    """
    LLaMA model class
    """
    def __init__(self, cfg):
        """
        Initialize the LLaMA model with the given config.

        Args:
            cfg: The LLaMA model configuration.
        """
        super().__init__()
        self.cfg = cfg
        self.token_embd = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        self.transformer_blocks = nn.Sequential(
            *[LlamaTransformer(cfg) for _ in range(cfg.n_layers)]
        )
        self.rms = RMSNormalization(cfg.embed_dim)
        self.output_layer = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False, dtype=cfg.dtype)
    
    def forward(self, x, targets=None):
        """
        Forward pass of the LLaMA model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): The target tensor for computing loss, of shape (batch_size, sequence_length).

        Returns:
            logits (torch.Tensor): The output logits of shape (batch_size, sequence_length, vocab_size).
            loss (torch.Tensor): The cross-entropy loss between the output logits and the target tensor.
        """
        x = self.token_embd(x) # Embed the input tokens
        x = self.transformer_blocks(x) # Pass the embedded input through the transformer blocks
        x = self.rms(x) # Apply layer normalization
        logits = self.output_layer(x) # Project the output to vocabulary size
        loss = None
        if targets is not None:
            # Compute the cross-entropy loss between the output logits and the target tensor
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # Reshape logits for cross-entropy
                targets.view(-1), # Reshape targets to match logits
                reduction="mean" # Compute mean loss
            )
        return logits, loss
    
    def generate(self, text, tokenizer, max_new_tokens=50, temperature=1.0, top_k=50):
        """
        Generate text using the model.

        Args:
            text (str): The input prompt text.
            tokenizer (Tokenizer): The tokenizer instance to encode and decode text.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature for randomness.
            top_k (int): If > 0, restrict sampling to the top k tokens.

        Returns:
            str: The generated text.
        """

        self.eval()
        
        # Encode the input text to token IDs
        input_ids = tokenizer.encode(text, bos=True, eos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=next(self.parameters()).device)

        for _ in range(max_new_tokens):
            # Get model output
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_values)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample a token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the token to the input sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == tokenizer.eos_id:
                break

        # Decode the generated token IDs back to text
        output_ids = input_ids[0].tolist()
        generated_text = tokenizer.decode(output_ids)

        return generated_text


# cfg = Llama31_CONFIG_8B()
# tokenizer = Tokenizer("Llama 3.1 8B tokenizer.model")
# model = Llama(cfg)
# model.tokenizer = tokenizer  # Make sure to set the tokenizer as an attribute

# prompt = "Once upon a time"
# generated_text = model.generate(
#     text=prompt,
#     tokenizer=tokenizer,
#     max_new_tokens=1,
#     temperature=0.8,
#     top_k=4,
# )
# print("Generated Text:", generated_text)

