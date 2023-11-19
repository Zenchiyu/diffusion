# Adapted from template made by Eloi Alonso
import torch
import torch.nn as nn
from typing import Optional
from blocks import NoiseEmbedding, LabelEmbedding, ResidualBlock, CondResidualBlock


class ResNet(nn.Module):
    def __init__(self,
                 image_channels: int,
                 nb_channels: int,
                 num_blocks: int,
                 cond_channels: int,
                 use_cond: bool=False) -> None:
        super().__init__()
        self.use_cond = use_cond
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.label_emb = LabelEmbedding(cond_channels)  # TODO: to put this in comment if try to load old model

        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)

        self.blocks  = nn.ModuleList([ResidualBlock(nb_channels) if not(use_cond) else
                                     CondResidualBlock(nb_channels, cond_channels)
                                     for _ in range(num_blocks)])
        
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
        cond += self.label_emb(c_label) if c_label is not None else 0  # The Karras et al. paper also adds embeddings (see p. 46)
        kwargs = {"cond": cond} if self.use_cond else {}
        
        ## Forward w/ or w/o conditioning
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, **kwargs)
        return self.conv_out(x)