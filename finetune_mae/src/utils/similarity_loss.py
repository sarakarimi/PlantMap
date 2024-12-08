import torch
import torch.nn as nn


class SimilarityLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # Concatenate both augmented views
        similarity_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0))
        mask = torch.eye(batch_size * 2, device=z.device).bool()
        positive_sim = torch.cat(
            [
                similarity_matrix[i, i + batch_size].unsqueeze(0)
                for i in range(batch_size)
            ]
            + [
                similarity_matrix[i + batch_size, i].unsqueeze(0)
                for i in range(batch_size)
            ]
        )
        similarity_matrix[mask] = -float("inf")
        exp_sim = torch.exp(similarity_matrix / self.temperature)
        loss = -torch.log(positive_sim / exp_sim.sum(dim=-1)).mean()
        return loss
