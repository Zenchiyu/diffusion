import torch
import torch.nn as nn
from enum import Enum
from typing import Optional
from .blocks import NoiseEmbedding, LabelEmbedding, CondUpDownBlock, State


class UNet(nn.Module):
    def __init__(self,
                 image_channels: int,
                 in_channels: int,
                 mid_channels: int,
                 nb_blocks: int,
                 cond_channels: int,
                 nb_classes: Optional[int]=None):
        super().__init__()
        assert (nb_blocks % 2 == 0) and (nb_blocks >= 2)
        self.nb_classes = nb_classes
        self.noise_emb = NoiseEmbedding(cond_channels)
        if self.nb_classes: self.label_emb = LabelEmbedding(nb_classes+1, cond_channels)  # incl. fake label to represent uncond.

        self.conv_in = nn.Conv2d(image_channels, in_channels, kernel_size=3, padding=1)
        down_blocks  = []
        up_blocks  = []
        for i in range(nb_blocks//2+1):  # +1 due to the bridge
            nic = in_channels if i == 0 else mid
            mid = mid_channels if i == 0 else nic*2
            updown_state = State.NONE if i == 0 else State.DOWN

            down_blocks.append(CondUpDownBlock(in_channels=nic,
                                                mid_channels=mid,
                                                out_channels=mid,
                                                cond_channels=cond_channels,
                                                updown_state=updown_state))
        # # Bridge
        # nic = mid
        # mid = nic*2
        # self.bridge = CondUpDownBlock(in_channels=nic,
        #                               mid_channels=mid,
        #                               out_channels=mid,
        #                               cond_channels=cond_channels,
        #                               updown_state=State.DOWN)
        for i in range(nb_blocks//2):
            nic = 2*mid_channels if i == nb_blocks//2-1 else mid
            mid = in_channels if i == nb_blocks//2-1 else nic//2
            
            up_blocks.append(CondUpDownBlock(in_channels=nic,
                                                mid_channels=mid,
                                                out_channels=mid,
                                                cond_channels=cond_channels,
                                                updown_state=State.UP))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_out = nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)
        
    def forward(self,
                noisy_input: torch.Tensor,
                c_noise: torch.Tensor,
                c_label: Optional[torch.Tensor]=None) -> torch.Tensor:
        ## Conditioning
        cond = self.noise_emb(c_noise)
        # Classifier-Free Guidance
        cond += self.label_emb(c_label) if (c_label is not None and self.nb_classes is not None) else 0
        
        ## Forward w/ conditioning
        x = self.conv_in(noisy_input)
        skips = []
        for block in self.down_blocks[:-1]:
            x = block(x, cond)
            skips.append(x)
        x = self.down_blocks[-1](x, cond)
        for skip, block in zip(reversed(skips), self.up_blocks):
            x = block(x, cond, skip)
        return self.conv_out(x)