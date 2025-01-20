import torch
import torch.nn as nn


class SimilarityLoss(nn.Module):
    """
    Helper class to compute the loss for the contrastive learning.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(z, z.t())  # Shape: (2*batch_size, 2*batch_size)
        
        # Extract positive pairs (before masking)
        positive_indices = torch.arange(batch_size, device=z.device)
        positive_sim = torch.cat([
            similarity_matrix[positive_indices, positive_indices + batch_size],
            similarity_matrix[positive_indices + batch_size, positive_indices]
        ])
        
        # Mask out self-similarities (diagonal elements)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        similarity_matrix.masked_fill_(mask, -float("inf"))
        
        # Compute log-softmax denominator
        exp_sim = torch.exp(similarity_matrix / self.temperature)
        denominator = exp_sim.sum(dim=1)
        
        # Loss calculation
        loss = -torch.log(torch.exp(positive_sim / self.temperature) / denominator).mean()
        return loss
