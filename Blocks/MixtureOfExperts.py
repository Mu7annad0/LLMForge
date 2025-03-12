import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """A layer that implements an expert network using a feedforward neural network."""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    """Router that selects top-k experts based on noisy logits."""
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        #Get top k logits
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    """Sparse mixture of experts model that routes inputs to selected experts."""
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([ExpertLayer(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
    
    def forward(self, x):
    # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))  
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)

        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(batch_size, seq_len, -1)
        return final_output


def main():
    """Main function to run the SparseMoE model with sample parameters."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    n_embed = 128   # Embedding dimension
    num_experts = 4 # Number of experts
    top_k = 2       # Number of experts to route to for each token
    batch_size = 3  # Batch size
    seq_len = 5     # Sequence length
    capacity_factor = 1.5  # Capacity factor (>1 means extra capacity)
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, n_embed)
    print(f"Input shape: {x.shape}")
    
    # Create the MoE model
    moe = SparseMoE(n_embed, num_experts, top_k, capacity_factor)
    
    # Calculate and print expert capacity
    tokens_per_batch = batch_size * seq_len * top_k
    expert_capacity = int((tokens_per_batch / num_experts) * capacity_factor)
    print(f"Number of experts: {num_experts}")
    print(f"Top-k: {top_k}")
    print(f"Total tokens: {batch_size * seq_len}")
    print(f"Total dispatches: {tokens_per_batch} (tokens * top_k)")
    print(f"Expert capacity: {expert_capacity} tokens per expert")
    
    # Print routing statistics
    with torch.no_grad():
        # Forward pass
        _, indices = moe.router(x)
        
        # Print expert usage
        print("\nExpert routing statistics:")
        for i in range(num_experts):
            expert_mask = (indices == i).any(dim=-1)
            tokens_routed = expert_mask.sum().item()
            print(f"Expert {i}: {tokens_routed}/{batch_size * seq_len} tokens " +
                  f"({tokens_routed/(batch_size * seq_len)*100:.1f}%)")
        
        # Forward pass to get output
        output = moe(x)
        print(f"\nOutput shape: {output.shape}")
        
        # Test if capacity limits are being enforced
        limited_experts = []
        for i in range(num_experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            if selected_indices.numel() > expert_capacity:
                limited_experts.append(i)
        
        if limited_experts:
            print(f"\nExperts with tokens dropped due to capacity limits: {limited_experts}")
        else:
            print("\nNo experts reached capacity limits")


if __name__ == "__main__":
    main()