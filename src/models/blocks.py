import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Optional, Callable


State = Enum('State', ['UP', 'DOWN', 'NONE'])

# CondModule & CondSequential: Helper classes to make code cleaner
class CondModule(nn.Module):
    pass

class CondResSeq(nn.Sequential, CondModule):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.process = layers[0]
        self.layers = layers[1:]

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor,
                skip: Optional[torch.Tensor]=None) -> torch.Tensor:
        y = self.process(x)
        y = y if skip is None else torch.cat([y, skip], dim=1)  # channel-wise
        for layer in self.layers:
            if isinstance(layer, CondModule):
                y = layer(y, cond)
            else:
                y = layer(y)
        return y

class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        # Outer-product
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight     # input.shape[0] x cond_channels//2
        return torch.cat([f.cos(), f.sin()], dim=-1)            # input.shape[0] x cond_channels
    
class LabelEmbedding(nn.Module):
    def __init__(self,
                 nb_classes: int,
                 cond_channels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(nb_classes, cond_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)

class CondBatchNorm2d(CondModule):
    def __init__(self,
                 nb_channels: int,
                 cond_channels: int,
                 special_init: bool=True) -> None:
        super().__init__()
        self.bn_params = nn.Linear(cond_channels, 2*nb_channels)
        self.norm = nn.BatchNorm2d(nb_channels, affine=False)
        # Init gamma, beta s.t. y=bn(x)=norm(x) behavior at init.
        if special_init:
            nn.init.zeros_(self.bn_params.weight)
            nn.init.zeros_(self.bn_params.bias)

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.bn_params(cond)[:, :, None, None].chunk(2, dim=1)  # N x C x 1 x 1 each
        return beta + (gamma+1)*self.norm(x)
    
class CondResidualBlock(CondModule):
    def __init__(self,
                 in_channels: int,
                 cond_channels: int,
                 mid_channels: Optional[int]=None,
                 out_channels: Optional[int]=None,
                 special_init: bool=True) -> None:
        super().__init__()
        mid_channels, out_channels = mid_channels or in_channels, out_channels or in_channels

        self.norm1 = CondBatchNorm2d(in_channels, cond_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm2 = CondBatchNorm2d(mid_channels, cond_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

        if out_channels != in_channels: 
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # spatial-resolution unchanged
        
        # Init s.t. y=x or y=proj(x) at init. where proj initially projects
        # each Cin x 1 x 1 activation into Cout x 1 x 1 using orthogonal vects.
        if special_init:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)
            nn.init.orthogonal_(self.proj.weight)

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x, cond)))
        y = self.conv2(F.silu(self.norm2(y, cond)))
        if x.shape != y.shape:
            return self.proj(x) + y
        return x + y

class MHSelfAttention2d(CondModule):
    def __init__(self,
                 in_channels: int,
                 nb_heads: int,
                 norm: Callable[[int], nn.Module],
                 special_init: bool=True) -> None:
        super().__init__()
        assert in_channels % nb_heads == 0, "q,k,v emb. dim: in_channels/nb_heads should be an integer"
        self.nb_heads, self.norm = nb_heads, norm(in_channels)
        self.qkv_proj = nn.Conv2d(in_channels, in_channels*3, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # shape unchanged because already correct
        # Init s.t. y=x at init.
        if special_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(self.norm(x, cond))                                        # N x 3C x H x W
        qkv = qkv.view(x.shape[0], 3*self.nb_heads, x.shape[1]//self.nb_heads, -1)     # N x 3nb_heads x C//nb_heads x HW
        Q, K, V = (qkv.transpose(-1, -2)).chunk(3, dim=1)                              # N x nb_heads x HW x C//nb_heads each
        Y = F.scaled_dot_product_attention(Q, K, V)                                    # N x nb_heads x HW x C//nb_heads
        y = Y.transpose(-1, -2).contiguous().view(*x.shape)                            # N x C x H x W
        return x + self.out_proj(y)

class CondUpDownBlock(CondResSeq):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 cond_channels: int,
                 nb_heads: int=2,
                 nb_layers: int=1,
                 self_attention: bool=True,
                 updown_state: State=State.NONE) -> None:
        super().__init__()
        # All the conditioning go into the normalization layers
        self.updown_state = updown_state
        self.layers = nn.ModuleList([])
        mid = mid_channels

        # Downsampling/avg pooling, nb of channels stay the same
        # Karras used a conv layer with fixed kernel
        if updown_state == State.DOWN:
            self.layers.append(nn.AvgPool2d(2, 2))
        # Upsampling + half the num. of channels via conv.
        # Karras used a conv layer with fixed kernel
        elif updown_state == State.UP:
            self.layers.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                             nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)))
        
        for i in range(nb_layers):
            nic = in_channels if i == 0 else mid
            noc = out_channels if i == nb_layers-1 else mid
            norm = lambda nb_channels: CondBatchNorm2d(nb_channels, cond_channels)  # a different BN per layer
            
            self.layers.append(CondResidualBlock(nic, mid, noc, cond_channels))
            if self_attention:
                self.layers.append(MHSelfAttention2d(in_channels=noc, nb_heads=nb_heads, norm=norm))
        
    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor,
                skip: Optional[torch.Tensor]=None) -> torch.Tensor:
        return super().forward(x, cond, skip)