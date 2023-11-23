import torch
import torch.nn as nn
from typing import Optional
from .blocks import NoiseEmbedding, LabelEmbedding, UpDownBlock


class UNet(nn.Module):
    def __init__(self,
                 image_channels: int,
                 in_channels: int,
                 mid_channels: int,
                 num_blocks: int,
                 cond_channels: int,
                 num_classes: Optional[int]=None):
        super().__init__()
        self.num_classes = num_classes
        self.noise_emb = NoiseEmbedding(cond_channels)
        if self.num_classes: self.label_emb = LabelEmbedding(num_classes+1, cond_channels)  # incl. fake label to represent uncond.

        self.conv_in = nn.Conv2d(image_channels, in_channels, kernel_size=3, padding=1)
        down_blocks  = []
        up_blocks  = []
        for i in range(num_blocks):
            nic = in_channels if i == 0 else mid_channels*2**i
            mid = mid_channels if i == 0 else nic*2

            down_blocks.append(UpDownBlock(in_channels=nic,
                                                mid_channels=mid,
                                                out_channels=mid,
                                                cond_channels=cond_channels,
                                                down=True))
        # Bridge
        up_blocks.append(UpDownBlock(in_channels=mid,
                                     mid_channels=mid*2,
                                     out_channels=mid*2,
                                     cond_channels=cond_channels,
                                     down=False))
        for i in range(num_blocks):
            nic = mid_channels if i == 0 else mid*2
            mid = in_channels if i == 0 else mid_channels*2**i
            
            up_blocks.append(UpDownBlock(in_channels=nic,
                                                mid_channels=mid,
                                                out_channels=mid,
                                                cond_channels=cond_channels,
                                                down=False))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(reversed(up_blocks))
        self.conv_out = nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)
        
    def forward(self,
                noisy_input: torch.Tensor,
                c_noise: torch.Tensor,
                c_label: Optional[torch.Tensor]=None) -> torch.Tensor:
        ## Conditioning
        cond = self.noise_emb(c_noise)
        # Classifier-Free Guidance
        cond += self.label_emb(c_label) if (c_label is not None and self.num_classes is not None) else 0
        
        ## Forward w/ conditioning
        x = self.conv_in(noisy_input)
        skips = []
        for block in self.down_blocks:
            x = block(x, cond)
            skips.append(x)
        skips.append(None)  # bridge
        for skip, block in zip(reversed(skips), self.up_blocks):
            x = block(x, cond, skip=skip)
        return self.conv_out(x)