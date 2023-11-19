# Adapted from template made by Eloi Alonso
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
        # Outer-product; f is of shape input.shape[0] x cond_channels//2
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        # Output is of shape input.shape[0] x cond_channels where
        # first half of the columns use cos and the other half use sin.
        return torch.cat([f.cos(), f.sin()], dim=-1)  # concat along last dim
    
class LabelEmbedding(nn.Module):
    # XXX: Same as NoiseEmbedding for the moment
    # The Karras et al. paper seemed to have used a MLP instead (see p. 46-47)
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BN -> ReLU -> Conv instead of Conv -> BN -> ReLU
        y = self.conv1(F.relu(self.norm1(x)))
        y = self.conv2(F.relu(self.norm2(y)))
        return x + y  # residual connection
    
class CondResidualBlock(nn.Module):
    def __init__(self,
                 nb_channels: int,
                 cond_channels: int) -> None:
        super().__init__()
        self.bn_params1 = nn.Linear(cond_channels, 2*nb_channels)
        self.bn_params2 = nn.Linear(cond_channels, 2*nb_channels)
        
        self.norm1 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        nb_channels = x.shape[1]
        # N x C' x 1 x 1 because different noises
        shape = (x.shape[0], 2*nb_channels, 1, 1)
        # Predict BN parameters using MLPs taking as input the noise embedding
        # N x C x 1 x 1 each
        gamma1, beta1 = self.bn_params1(cond).view(*shape).split(nb_channels, dim=1)
        gamma2, beta2 = self.bn_params2(cond).view(*shape).split(nb_channels, dim=1)

        # BN -> ReLU -> Conv instead of Conv -> BN -> ReLU
        bn1 = torch.addcmul(beta1, gamma1, self.norm1(x))
        y = self.conv1(F.relu(bn1))
        bn2 = torch.addcmul(beta2, gamma2, self.norm2(y))
        y = self.conv2(F.relu(bn2))
        return x + y  # residual connection
