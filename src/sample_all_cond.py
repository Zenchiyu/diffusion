import hydra
import torch

from init import init
from sampler import sample as sample_helper
from utils import save

from omegaconf import DictConfig
from pathlib import Path
from typing import Optional


def sample(nb_chunks=2, **kwargs) -> torch.Tensor:
    assert kwargs["num_samples"] % nb_chunks == 0,\
        "num_samples should be a multiple of the number of chunks"
    num_samples = kwargs["num_samples"]//nb_chunks
    label = kwargs.get("label", None)
    chunked_labels = label.chunk(nb_chunks) if label is not None else num_samples*[None]
    
    def new_kwargs(label: Optional[torch.Tensor]=None) -> torch.Tensor:
        return kwargs| {"num_samples": num_samples, "label": label}
    
    return torch.cat([sample_helper(**new_kwargs(label)) for label in chunked_labels], dim=0)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def sample_all_cond(cfg: DictConfig):
    seed = torch.random.initial_seed()  # retrieve current seed

    # Initialization
    init_tuple = init(cfg)
    model, diffusion, info = init_tuple.model, init_tuple.diffusion, init_tuple.info
    try:
        sampling_method = cfg.common.sampling.method
    except:
        sampling_method = "euler"
    dataset_name = str.lower(cfg.dataset.name)
    cfgscale_str = str(cfg.common.sampling.cfg_scale).replace('.','_')
    path = Path(f"./results/images/{dataset_name}/{sampling_method}/")
    
    # Don't use the checkpoint seed for sampling
    torch.manual_seed(seed)

    # Sample 90 pictures for each class and save
    nb_per_class = 3*30
    N, C, H, W = nb_per_class*info.num_classes, info.image_channels, info.image_size, info.image_size
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
    samples = sample(**kwargs)

    save(samples.view(info.num_classes, nb_per_class, C, H, W)[:, :10].reshape(-1, C, H, W),
         path / f"cond_10_cfgscale_{cfgscale_str}.png", nrow=10, padding=1)
    save(samples, path / f"cond_{nb_per_class}_cfgscale_{cfgscale_str}.png", nrow=30, padding=1)

if __name__ == "__main__":
    sample_all_cond()
