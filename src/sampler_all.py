import hydra
import torch

from init import init
from utils import save

from omegaconf import DictConfig
from sampler import sample

@hydra.main(version_base=None, config_path="../config", config_name="config")
def sampler(cfg: DictConfig):
    # Initialization
    init_tuple = init(cfg)
    model, diffusion, info = init_tuple.model, init_tuple.diffusion, init_tuple.info

    # Don't use the checkpoint seed for sampling
    torch.seed()

    # Sample 90 pictures for each class and save
    nb_per_class = 3*30
    N, C, H, W = nb_per_class*info.num_classes, info.image_channels, info.image_size, info.image_size
    samples = sample(
            N, C, H, model, diffusion,
            uncond_label=info.num_classes,
            label=torch.repeat_interleave(torch.arange(info.num_classes), nb_per_class),
            cfg_scale=cfg.common.sampling.cfg_scale,
            num_steps=cfg.common.sampling.num_steps)
    dataset_name = str.lower(cfg.dataset.name)
    cfgscale_str = str(cfg.common.sampling.cfg_scale).replace('.','_')

    save(samples.view(info.num_classes, nb_per_class, C, H, W)[:, :10].reshape(-1, C, H, W),
        f"./src/images/all_{dataset_name}_10_cfgscale_{cfgscale_str}.png",
        nrow=10, padding=1)
    save(samples,
        f"./src/images/all_{dataset_name}_90_cfgscale_{cfgscale_str}.png",
        nrow=30, padding=1)

if __name__ == "__main__":
    sampler()
