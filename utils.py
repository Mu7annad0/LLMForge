import torch
import tiktoken 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    This function generates text using a simple method. 
    It takes a model, an index sequence, the maximum number of new tokens to generate, and the size of the context.
    """
    # Loop to generate new tokens up to max_new_tokens
    for _ in range(max_new_tokens):
        # Extract the last context_size tokens from the current index sequence
        idx_cond = idx[:, -context_size:]
        # Run the model in inference mode to generate logits for the next token
        with torch.no_grad():
            logits = model(idx_cond)

        # Extract the logits for the last token in the sequence
        logits = logits[:, -1, :]
        # Convert logits to probabilities
        probas = torch.softmax(logits, dim=-1)
        # Select the token with the highest probability as the next token
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # Append the new token to the index sequence
        idx = torch.cat((idx,idx_next), dim=-1)

    # Return the updated index sequence with new tokens
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # Convert the encoded token IDs into a tensor and add a batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(ids, tokenizer):
    # Remove the batch dimension from the tensor of token IDs
    decoded = ids.squeeze(0)
    return tokenizer.decode(decoded.tolist())

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)   
    else:
        num_batches = min(num_batches, len(data_loader))   
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)      
            # Forward pass to get logits
            logits = model(input_batch)
            # Calculate loss using cross-entropy
            loss = torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1), target_batch.flatten()
                    )
            total_loss += loss.item()  
        else:
            break
    return total_loss / num_batches


def generate_and_print_text(model, tokenizer, device, start_context):
    model.eval()
    # Determine the context size from the model's positional embedding
    contex_size = model.pos_embed.weight.shape[0]
    # Convert the start context into token IDs and move to the device
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # Generate text using the model
        token_ids = generate_text_simple(model, encoded, 50, contex_size)
    # Convert generated token IDs back into text
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
