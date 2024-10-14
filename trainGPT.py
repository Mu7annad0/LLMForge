import torch
import tiktoken
import os
import urllib.request
from dataclasses import dataclass

from Models.GPT import GPTModel
from utils import calc_loss_loader, generate_and_print_text, create_dataloader_v1, plot_losses


def train_model(model, tokenizer, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context):
    # Initialize lists to track training and validation losses, as well as tokens seen
    train_losses, val_losses, track_token_seen = [], [], []
    global_step = 0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Clear gradients
            input_batch = input_batch.to(device)  # Move input batch to the GPU
            target_batch = target_batch.to(device)  # Move target batch to the GPU
            logits = model(input_batch)  # Forward pass
            loss = torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1), target_batch.flatten()
                    )  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            global_step += 1  # Update global step

            if global_step % eval_freq == 0:  # Evaluate model at specified frequency
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, eval_iter)  # Calculate training loss
                    val_loss = calc_loss_loader(val_loader, model, device, eval_iter)  # Calculate validation loss
                model.train()  # Set the model back to training mode
                train_losses.append(train_loss)  # Append training loss to list
                val_losses.append(val_loss)  # Append validation loss to list

                print(f"EPOCH: {epoch+1} (step: {global_step:06d}) Train_loss: {train_loss:.3f} Val_loss: {val_loss:.3f}")  # Print training and validation loss
                
        generate_and_print_text(model, tokenizer, device, start_context)  # Generate and print text using the model
    
    return train_losses, val_losses  # Return training and validation losses, as well as tokens seen


def main(cfg, settings):
    torch.manual_seed(123)  # Set the seed for reproducibility

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Choose the device (MPS or CPU)

    model = GPTModel(GPT_CONFIG_124M)  # Initialize the model
    model.to(device)  # Move the model to the device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr = 0.0004, weight_decay=0.1  # Initialize the optimizer
    )
    num_epochs = 10  # Number of epochs
    tokenizer = tiktoken.get_encoding("gpt2")  # Initialize the tokenizer
    
    file_path = "the-verdict.txt"  # File path for the text data
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"  # URL for the text data

    if not os.path.exists(file_path):  # Check if the file exists
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')  # Download the text data
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)  # Write the text data to a file
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()  # Read the text data from a file

    train_ratio = 0.90  # Ratio for splitting the data into training and validation sets
    split_idx = int(train_ratio * len(text_data))  # Calculate the split index

    train_loader = create_dataloader_v1(
        text_data[:split_idx],  # Training data
        batch_size=settings.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],  # Validation data
        batch_size=settings.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    train_losses, val_losses = train_model(model, tokenizer, train_loader, val_loader,
                                                    optimizer, device, num_epochs, 5, 5, "Every effort moves you")
    
    return train_losses, val_losses
    

if __name__ == "__main__":

    @dataclass
    class GPT_CONFIG_124M:
        vocab_size: int =  50257
        context_length: int = 256
        embed_dim: int = 768
        n_heads: int = 12
        n_layers: int = 12
        drop_rate : float = 0.1
        qkv_bias : bool = False

    @dataclass
    class settings:
        learning_rate: float = 5e-4,
        num_epochs: int = 10,
        batch_size: int = 2,
        weight_decay: float = 0.1

    train_losses, val_losses = main(GPT_CONFIG_124M, settings)