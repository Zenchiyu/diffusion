# Initial template made by Eloi Alonso
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(nb_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise) # TODO: not used yet
        # Apply ResNet on noisy input
        # Compared to the embryo at: https://fleuret.org/dlc/src/dlc_practical_6_embryo.py
        # where we have:
        # Conv->BN->ReLU->manytimes([Conv->BN->ReLU->Conv->BN]->ReLU)->AvgPool->FC
        # This network has:
        # Conv->manytimes([BN->ReLU->Conv->BN->ReLU->Conv])->Conv
        # where [.] indicates a residual connection
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)


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
