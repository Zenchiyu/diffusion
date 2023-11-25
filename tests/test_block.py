import numpy as np
import torch
import torch.nn.functional as F
import unittest

from hydra import compose, initialize
from src.models.blocks import CondBatchNorm2d, NoiseEmbedding, LabelEmbedding
from src.models.blocks import CondResidualBlock, MHSelfAttention2d, CondUpDownBlock
from src.models.blocks import State

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
        self.inter_shape = (10, self.model_kwargs["in_channels"], 32, 32)
        self.label_shape = self.noise_shape  = (self.input_shape[0],)
        self.input = torch.randn(self.input_shape)
        self.inter = torch.randn(self.inter_shape)
        self.noise = torch.randn(self.input_shape[0])
        self.label = torch.randint(self.model_kwargs["nb_classes"], self.label_shape)

        self.noise_emb = NoiseEmbedding(self.model_kwargs["cond_channels"])
        if self.model_kwargs["nb_classes"]:
            self.label_emb = LabelEmbedding(self.model_kwargs["nb_classes"]+1,
                                            self.model_kwargs["cond_channels"])  # incl. fake label to represent uncond.
        ## Conditioning over noise and label
        self.cond = self.noise_emb(self.noise)
        self.cond += self.label_emb(self.label)

        # Number of times to repeat stochastic tests
        self.num_repeat = 7
        
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
        for _ in range(self.num_repeat):
            mid_channels, out_channels = np.random.randint(1, 10), np.random.randint(1, 10)*self.input_shape[1]
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
        for _ in range(self.num_repeat):
            # Input shape with random number of channels
            in_channels = 2**np.random.randint(5, 10+1)
            input_shape = (self.input_shape[0], in_channels, 32, 32)

            # Compute nb_heads that cuts evenly the nb of channels
            high = int(np.log2(in_channels))
            embed_dim = 2**np.random.randint(1, high+1)
            nb_heads = in_channels//embed_dim
            self.assertEqual(in_channels % nb_heads, 0, 'embed. dim. = in_channels/nb_heads should be an integer')

            # Compute nb_heads that doesn't evenly cut the nb of channels
            embed_dim += 1  # not a power of 2 anymore
            nb_heads = in_channels/embed_dim  # float

            norm = lambda nb_channels: CondBatchNorm2d(nb_channels, self.model_kwargs["cond_channels"])
            msg = 'MHSelfAttention2d should have a number of heads that '+\
                f'splits the channels evenly, i.e. in_channels/nb_heads={in_channels}/{nb_heads} should be an integer. '+\
                'Otherwise, it should raise an AssertionError'
            with self.assertRaises(AssertionError, msg=msg):
                block = MHSelfAttention2d(in_channels, nb_heads=nb_heads, norm=norm)
                y = block(torch.randn(input_shape), self.cond)
    
    def test_CondUpDownBlock_none_io_shapes(self):
        x = self.inter
        for _ in range(self.num_repeat):
            high = int(np.log2(self.inter_shape[1]))
            embed_dim = 2**np.random.randint(1, high+1) if high != 0 else 1
            nb_heads = self.inter_shape[1]//embed_dim
            out_channels, nb_layers = self.model_kwargs["in_channels"]*2, 1
            block = CondUpDownBlock(in_channels=self.model_kwargs["in_channels"],
                                    mid_channels=self.model_kwargs["mid_channels"],
                                    out_channels=out_channels,
                                    cond_channels=self.model_kwargs["cond_channels"],
                                    nb_heads=nb_heads,
                                    nb_layers=nb_layers,
                                    updown_state=State.NONE)
            y = block(x, self.cond, skip=None)
            self.assertNotEqual(y.shape[1], self.inter_shape[1], 'CondUpDownBlock State.NONE can change the number of channels')
            self.assertEqual((y.shape[0], *y.shape[2:]), (x.shape[0], *x.shape[2:]),
                            'CondUpDownBlock State.NONE can only change the number of channels.'+\
                            'Spatial resolution has to be kept unchanged')
            
    def test_CondUpDownBlock_down_io_shapes(self):
        x = self.inter
        for _ in range(self.num_repeat):
            high = int(np.log2(self.inter_shape[1]))
            embed_dim = 2**np.random.randint(1, high+1) if high != 0 else 1
            nb_heads = self.inter_shape[1]//embed_dim
            out_channels, nb_layers = self.model_kwargs["in_channels"]*2, 1
            block = CondUpDownBlock(in_channels=self.model_kwargs["in_channels"],
                                    mid_channels=self.model_kwargs["mid_channels"],
                                    out_channels=out_channels,
                                    cond_channels=self.model_kwargs["cond_channels"],
                                    nb_heads=nb_heads,
                                    nb_layers=nb_layers,
                                    updown_state=State.DOWN)
            y = block(x, self.cond, skip=None)
            self.assertNotEqual(y.shape[1], self.inter_shape[1], 'CondUpDownBlock State.DOWN can change the number of channels')
            self.assertEqual(y.shape[2:], (x.shape[-2]//2, x.shape[-1]//2),
                            'CondUpDownBlock State.DOWN halves the spatial dimension')
            
    def test_CondUpDownBlock_up_io_shapes(self):
        x = self.inter
        for _ in range(self.num_repeat):
            high = int(np.log2(self.inter_shape[1]))
            embed_dim = 2**np.random.randint(1, high+1) if high != 0 else 1
            nb_heads = self.inter_shape[1]//embed_dim
            out_channels, nb_layers = self.model_kwargs["in_channels"]*2, 1
            block = CondUpDownBlock(in_channels=self.model_kwargs["in_channels"],
                                    mid_channels=self.model_kwargs["mid_channels"],
                                    out_channels=out_channels,
                                    cond_channels=self.model_kwargs["cond_channels"],
                                    nb_heads=nb_heads,
                                    nb_layers=nb_layers,
                                    updown_state=State.UP)
            skip = torch.zeros((x.shape[0], x.shape[1]//2, 2*x.shape[2], 2*x.shape[3]))
            y = block(x, self.cond, skip=skip)
            self.assertNotEqual(y.shape[1], self.inter_shape[1], 'CondUpDownBlock State.UP can change the number of channels')
            self.assertEqual(y.shape[2:], skip.shape[2:],
                            'CondUpDownBlock State.UP doubles the spatial dimension')
        
    # TODO: test with multiple nb_layers

if __name__ == '__main__':
    unittest.main()