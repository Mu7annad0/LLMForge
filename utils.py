import torch

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