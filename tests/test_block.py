import numpy as np
import torch
import unittest

from hydra import compose, initialize
from src.models.blocks import CondBatchNorm2d, NoiseEmbedding, LabelEmbedding
from src.models.blocks import CondResidualBlock, MHSelfAttention2d


class TestBlocks(unittest.TestCase):
    def setUp(self):
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name='config.yaml')
        self.cfg = cfg
        self.model_kwargs = {
            "image_channels": 1,
            "in_channels": cfg.model.nb_channels,
            "mid_channels": cfg.model.nb_channels,
            "nb_blocks": cfg.model.num_blocks,
            "cond_channels": cfg.model.cond_channels,
            "nb_classes": 10
        }

        self.input_shape = (10, self.model_kwargs["image_channels"], 32, 32)
        self.label_shape = self.noise_shape  = (self.input_shape[0],)
        self.input = torch.randn(self.input_shape)
        self.noise = torch.randn(self.input_shape[0])
        self.label = torch.randint(self.model_kwargs["nb_classes"], self.label_shape)

        self.noise_emb = NoiseEmbedding(self.model_kwargs["cond_channels"])
        if self.model_kwargs["nb_classes"]:
            self.label_emb = LabelEmbedding(self.model_kwargs["nb_classes"]+1,
                                            self.model_kwargs["cond_channels"])  # incl. fake label to represent uncond.
        ## Conditioning over noise and label
        self.cond = self.noise_emb(self.noise)
        self.cond += self.label_emb(self.label)
        
    def test_CondBatchNorm2d_io_shapes(self):
        bn = CondBatchNorm2d(self.input_shape[1], self.model_kwargs["cond_channels"])
        y = bn(self.input, self.cond)
        self.assertEqual(y.shape, self.input_shape, 'CondBatchNorm2d should not change the input shape')

    def test_CondBatchNorm2d_stats(self):
        bn = CondBatchNorm2d(self.input_shape[1], self.model_kwargs["cond_channels"])
        torch.nn.init.zeros_(bn.bn_params.weight)
        torch.nn.init.ones_(bn.bn_params.bias)
        y = bn(self.input, self.cond)
        self.assertTrue(torch.allclose(y, bn.norm(self.input) + 1))

    def test_CondResidualBlock_io_shapes(self):
        mid_channels, out_channels = np.random.randint(10), np.random.randint(10)*self.input_shape[1]
        block = CondResidualBlock(self.input_shape[1], self.model_kwargs["cond_channels"],
                                  mid_channels, out_channels)
        y = block(self.input, self.cond)
        self.assertEqual(y.shape[0], self.input_shape[0], 'CondResidualBlock should keep the number of samples')
        self.assertEqual(y.shape[1], out_channels, 'CondResidualBlock can change the number of channels')
        self.assertEqual(y.shape[-2:], self.input_shape[-2:], 'CondResidualBlock should keep the spatial resolution')
    
    def test_MHSelfAttention2d_io_shapes(self):
        norm = lambda nb_channels: CondBatchNorm2d(nb_channels, self.model_kwargs["cond_channels"])
        block = MHSelfAttention2d(self.input_shape[1], nb_heads=1, norm=norm)
        y = block(self.input, self.cond)
        self.assertEqual(y.shape, self.input_shape, 'MHSelfAttention2d should not change the input shape')

    def test_MHSelfAttention2d_nb_heads(self):
        input_shape = (self.input_shape[0], 64, 32, 32)
        embed_dim = 2**np.random.randint(1, 5)
        nb_heads = input_shape[1]//embed_dim
        self.assertEqual(input_shape[1] % nb_heads, 0, 'embed. dim. = in_channels/nb_heads should be an integer')

        embed_dim += 1  # not a power of 2 anymore
        nb_heads = input_shape[1]//embed_dim

        norm = lambda nb_channels: CondBatchNorm2d(nb_channels, self.model_kwargs["cond_channels"])
        msg = 'MHSelfAttention2d should have a number of heads that '+\
              'splits the channels evenly, i.e. in_channels/nb_heads should be an integer.'+\
              'Otherwise, it should raise an AssertionError'
        with self.assertRaises(AssertionError, msg=msg):
            block = MHSelfAttention2d(input_shape[1], nb_heads=nb_heads, norm=norm)
            y = block(torch.randn(input_shape), self.cond)
        

if __name__ == '__main__':
    unittest.main()