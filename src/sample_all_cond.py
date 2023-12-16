import hydra
import torch

from init import init_sampling
from sampler import sample_chunked as sample
from utils import save

from omegaconf import DictConfig

@torch.no_grad()
@hydra.main(version_base=None, config_path="../config", config_name="config")
def sample_all_cond(cfg: DictConfig):
    model, diffusion, info, _, sampling_method, path, cfgscale_str = init_sampling(cfg)
    
    # Sample 90 pictures for each class and save
    nb_per_class = 3*30
    N, C, H, W = nb_per_class*info.num_classes, info.image_channels, 64, 64 # info.image_size, info.image_size
    kwargs = {
        "num_samples": N,
        "image_channels": C,
        "image_size": H,
        "model": model,
        "diffusion": diffusion,
        "uncond_label": info.num_classes,
        "label": torch.repeat_interleave(torch.arange(info.num_classes), nb_per_class),
        "cfg_scale": cfg.common.sampling.cfg_scale,
        "num_steps": cfg.common.sampling.num_steps,
        "sampling_method": sampling_method,
    }
    samples = sample(nb_chunks=30,**kwargs)

    save(samples.view(info.num_classes, nb_per_class, C, H, W)[:, :10].reshape(-1, C, H, W),
         path / f"cond_10_cfgscale_{cfgscale_str}_64x64.png", nrow=10, padding=1)
    save(samples, path / f"cond_{nb_per_class}_cfgscale_{cfgscale_str}_64x64.png", nrow=30, padding=1)

if __name__ == "__main__":
    sample_all_cond()
