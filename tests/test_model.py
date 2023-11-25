import torch
import unittest

from hydra import compose, initialize
from pathlib import Path
from src.models.unet import UNet


class TestUNet(unittest.TestCase):
    def setUp(self):
        # Initialization
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
        self.model = UNet(**self.model_kwargs)

        self.input_shape = (10, self.model_kwargs["image_channels"], 32, 32)
        self.label_shape = self.noise_shape  = (self.input_shape[0],)
        self.input = torch.randn(self.input_shape)
        self.noise = torch.randn(self.input_shape[0])
        self.label = torch.randint(self.model_kwargs["nb_classes"], self.label_shape)
    
    @unittest.skip("demonstrating skipping")
    def test_output_shape(self):
        output = self.model(self.input, self.noise, self.label)
        self.assertEqual(output.shape == self.input_shape)

if __name__ == '__main__':
    unittest.main()