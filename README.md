# LLM FORGE (Building Large Language Models from Scratch)

This project is a collection of my implementations of popular language models and attention mechanisms, including GPT-2 and LLaMA architectures, built using PyTorch.

## Project Structure

```
├── Attentions/                  # Attention mechanism implementations
│   └── GroupedQueryAttention.py # Implementation of Grouped Query Attention
|   └── CausalAttention.py       # Implementation of Causal Attetnion
|   └── SimpleSelfAttetnion.py   # Implementation of Simple Attention layer
|
├── Blocks/                      # Core building blocks for the models
|   ├── Activations.py           # Activation functions (GELU, SiLU)
│   ├── Configs.py               # Model configurations
|   ├── FeedForwards.py          # Feed forward blocks for GPT and Llama models
│   ├── Normalizations.py        # Layer and RMS Normalization implementations
|   ├── Positionals.py           # Positonal encoding and RoPE implementations
│   └── Transformers.py          # Transformer blocks for GPT2 and Llama
|
├── Models/                      # Main model implementations
│   ├── GPT2.py                  # GPT-2 model implementation
│   └── Llama.py                 # LLaMA model implementation
|
├── tokenizer.py                 # Llama 3 tokenizer implementation
└── trainGPT2.py                 # Training script for GPT-2 model
```

## Features

- Implementation of GPT-2 architecture
- Implementation of LLaMA architecture
- Custom attention mechanisms including Grouped Query Attention, Causal Attention
- Modular design with separate blocks for transformers, normalizations, Postionals, and configurations
- Training script for GPT-2 model


## Training GPT-2

1. Install Dependencies:
   ```bash
   pip install -r requirements.txt

2. Train the Model:
    ```bash
    python -m trainGPT2

## Future Work

1. **LLaMA Training Implementation**
   - Develop a comprehensive training pipeline for the LLaMA model
   - Add support for distributed training across multiple GPUs

2. **Model Fine-tuning**
   - Implement fine-tuning capabilities for specific tasks
   - Support for LoRA and other efficient fine-tuning methods

3. **LLM Research Implementation**
   - Implement latest research papers in the field of LLMs
   - Implement different attention mechanisms
