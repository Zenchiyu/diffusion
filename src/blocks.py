import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        # Outer-product; input.shape[0] x cond_channels//2
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        # input.shape[0] x cond_channels by concat along last dim
        return torch.cat([f.cos(), f.sin()], dim=-1)
    
class LabelEmbedding(nn.Module):
    def __init__(self,
                 num_classes: int,
                 cond_channels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, cond_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)

class CondBatchNorm2d(nn.Module):
    def __init__(self,
                 nb_channels: int,
                 cond_channels: int) -> None:
        super().__init__()
        self.bn_params = nn.Linear(cond_channels, 2*nb_channels)
        self.norm = nn.BatchNorm2d(nb_channels, affine=False)

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor) :
        # N x C x 1 x 1 each
        gamma, beta = self.bn_params(cond)[:, :, None, None].chunk(2, dim=1)
        return beta+gamma*self.norm(x)
    
class CondResidualBlock(nn.Module):
    def __init__(self,
                 nb_channels: int,
                 cond_channels: int) -> None:
        super().__init__()
        self.norm1 = CondBatchNorm2d(nb_channels, cond_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = CondBatchNorm2d(nb_channels, cond_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        # BN -> ReLU -> Conv instead of Conv -> BN -> ReLU
        y = self.conv1(F.relu(self.norm1(x, cond)))
        y = self.conv2(F.relu(self.norm2(y, cond)))
        return x + y  # residual connection
