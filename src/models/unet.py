import torch
import torch.nn as nn
from typing import Optional
from .blocks import NoiseEmbedding, LabelEmbedding, CondUpDownBlock, State


class UNet(nn.Module):
    def __init__(self,
                 image_channels: int,
                 in_channels: int,
                 min_channels: int,
                 depths: list[int],
                 cond_channels: int,
                 self_attentions: bool|list[bool] = True,
                 self_attention_bridge: bool=True,
                 nb_classes: Optional[int]=None):
        super().__init__()
        assert (len(depths) >= 1)
        
        if isinstance(self_attentions, bool):
            self_attentions = [self_attentions]*len(depths)

        self.nb_classes = nb_classes
        self.noise_emb = NoiseEmbedding(cond_channels)
        if self.nb_classes: self.label_emb = LabelEmbedding(nb_classes+1, cond_channels)  # incl. fake label to represent uncond.

        self.conv_in = nn.Conv2d(image_channels, in_channels, kernel_size=3, padding=1)
        down_blocks  = []
        up_blocks  = []
        for i in range(len(depths)+1):  # +1 due to the bridge
            nic = in_channels if i == 0 else mid
            mid = min_channels if i == 0 else nic*2
            updown_state = State.NONE if i == 0 else State.DOWN
            self_attention = self_attentions[i] if i != len(depths) else self_attention_bridge
            idx = min(i, len(depths)-1)
            down_blocks.append(CondUpDownBlock(in_channels=nic,
                                                mid_channels=mid,
                                                out_channels=mid,
                                                cond_channels=cond_channels,
                                                nb_layers=depths[idx],
                                                self_attention=self_attention,
                                                updown_state=updown_state))
        for i in range(len(depths)):
            nic = 2*min_channels if i == len(depths)-1 else mid
            mid = in_channels if i == len(depths)-1 else nic//2
            idx = len(depths)-i-1
            
            up_blocks.append(CondUpDownBlock(in_channels=nic,
                                                mid_channels=mid,
                                                out_channels=mid,
                                                cond_channels=cond_channels,
                                                nb_layers=depths[idx],
                                                self_attention=self_attentions[idx],
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