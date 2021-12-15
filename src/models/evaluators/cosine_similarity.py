import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, dim=1):
        return F.cosine_similarity(x1, x2, dim=dim).mean().item()