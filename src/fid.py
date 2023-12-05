import hydra
import torch

from tqdm import tqdm
from torcheval.metrics import FrechetInceptionDistance
from init import init
from omegaconf import DictConfig
from sampler import sample
from utils import float2tensor


@hydra.main(version_base=None, config_path="../config", config_name="config")
def fid(cfg: DictConfig):
    seed = torch.random.initial_seed()  # retrieve current seed

    # Initialization
    init_tuple = init(cfg)
    model, diffusion, info, device, dl = init_tuple.model, init_tuple.diffusion, init_tuple.info, init_tuple.device, init_tuple.dl

    # Don't use the checkpoint seed for sampling
    torch.manual_seed(seed)

    # Sample and display
    try:
        sampling_method = cfg.common.sampling.method
    except:
        sampling_method = "euler"

    batch_size = 10
    N, C, H, W = batch_size, info.image_channels, info.image_size, info.image_size

    metric = FrechetInceptionDistance(device=device)
    for (images, _) in dl.test:
        metric.update(float2tensor(images), is_real=True)

    for _ in tqdm(range(0, 10_000, batch_size)):  # 50_000
        fake_samples = sample(
                            N, C, H, model, diffusion,
                            uncond_label=info.num_classes,
                            label=cfg.common.sampling.label,
                            cfg_scale=cfg.common.sampling.cfg_scale,
                            num_steps=cfg.common.sampling.num_steps,
                            track_inter=False,
                            sampling_method=sampling_method
                            )
        metric.update(float2tensor(fake_samples), is_real=False)
    print(f"FID (test): {metric.compute()}")
    
if __name__ == "__main__":
    fid()
