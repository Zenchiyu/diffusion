import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable


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
                 in_channels: int,
                 cond_channels: int,
                 mid_channels: Optional[int]=None,
                 out_channels: Optional[int]=None) -> None:
        super().__init__()
        mid_channels, out_channels = mid_channels or in_channels, out_channels or in_channels

        self.norm1 = CondBatchNorm2d(in_channels, cond_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm2 = CondBatchNorm2d(mid_channels, cond_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

        if out_channels != in_channels:
            self.proj = nn.Conv2d(out_channels, in_channels, kernel_size=1)  # spatial-resolution unchanged
        # TODO: how important is the init here? Karras used nn.init.orthonogal_
    
    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x, cond)))
        y = self.conv2(F.relu(self.norm2(y, cond)))
        if x.shape != y.shape:
            return x + self.proj(y)
        return x + y

class MHSelfAttention2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 nb_heads: int,
                 norm: Callable[[int], nn.Module]) -> None:
        super().__init__()
        assert in_channels % nb_heads == 0, "q,k,v emb. dim: in_channels/nb_heads should be an integer"
        self.nb_heads, self.norm = nb_heads, norm(in_channels)
        self.qkv_proj = nn.Conv2d(in_channels, in_channels*3, kernel_size=1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # shape unchanged
        # TODO: how important is the init here? Karras set weights and biases of conv to 0 initially

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(self.norm(x, cond))             # N x 3C x H x W
        qkv = qkv.view(x.shape[0], 3*self.nb_heads, -1)     # N x 3nb_heads x C//nb_heads x HW
        Q, K, V = (qkv.transpose(-1, -2)).chunk(3, dim=1)   # N x nb_heads x HW x C//nb_heads each
        Y = F.scaled_dot_product_attention(Q, K, V)         # N x nb_heads x HW x C//nb_heads
        y = Y.transpose(-1, -2).contiguous().view(*x.shape) # N x C x H x W
        return x + self.conv(y)

class UpDownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,  # 'bottleneck'
                 out_channels: int,
                 cond_channels: int,
                 nb_heads: int=64,
                 num_layers: int=1,
                 down: Optional[bool]=None) -> None:
        super().__init__()
        # All the conditioning go into the normalization layers
        self.down = down
        self.layers = []
        mid = mid_channels

        # Downsampling/avg pooling to shrink the spatial resolution, not number of channels
        # TODO: Karras used something else
        if down: self.layers.append(nn.AvgPool2d(2, 2))

        for i in range(num_layers):
            nic = in_channels if i == 0 else mid
            noc = out_channels if i == num_layers-1 else mid

            norm = lambda nb_channels: CondBatchNorm2d(nb_channels, cond_channels)
            
            self.layers.append(CondResidualBlock(in_channels=nic,
                                                 mid_channels=mid,
                                                 out_channels=noc,
                                                 cond_channels=cond_channels))
            self.layers.append(MHSelfAttention2d(in_channels=noc,
                                                 nb_heads=nb_heads,
                                                 norm=norm))
        # Upsampling to increase the spatial resolution. Half also the num. of channels
        # TODO: Karras used something else
        if down is not None and not(down):
            self.layers.append(nn.Upsample(scale_factor=2))
            self.layers.append(nn.Conv2d(out_channels, out_channels//2, kernel_size=1, padding=1))
            # TODO: check the out_channels//2

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor,
                skip: Optional[torch.Tensor]=None) -> torch.Tensor:
        y = x if skip is None else torch.cat([x, skip], dim=1)  # channel-wise
        for l in self.layers:
            y = l(y, cond)
        return y
