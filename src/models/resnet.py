# Adapted from template made by Eloi Alonso
import torch
import torch.nn as nn
from typing import Optional
from .blocks import NoiseEmbedding, LabelEmbedding, CondResidualBlock


class ResNet(nn.Module):
    def __init__(self,
                 image_channels: int,
                 nb_channels: int,
                 num_blocks: int,
                 cond_channels: int,
                 num_classes: Optional[int]=None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.noise_emb = NoiseEmbedding(cond_channels)
        if self.num_classes: self.label_emb = LabelEmbedding(num_classes+1, cond_channels)  # incl. fake label to represent uncond.

        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks  = nn.ModuleList([CondResidualBlock(nb_channels, cond_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
        
    def forward(self,
                noisy_input: torch.Tensor,
                c_noise: torch.Tensor,
                c_label: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Apply ResNet on noisy input

        Compared to the embryo at: https://fleuret.org/dlc/src/dlc_practical_6_embryo.py
        where we have:
        Conv->BN->ReLU->manytimes([Conv->BN->ReLU->Conv->BN]->ReLU)->AvgPool->FC

        This network has:
        Conv->manytimes([BN->ReLU->Conv->BN->ReLU->Conv])->Conv
        where [.] indicates a residual connection
        """
        ## Conditioning
        cond = self.noise_emb(c_noise)
        # Classifier-Free Guidance
        cond += self.label_emb(c_label) if (c_label is not None and self.num_classes is not None) else 0
        
        ## Forward w/ conditioning
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)