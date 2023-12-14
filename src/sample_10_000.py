import hydra
import torch

from tqdm import tqdm
# from torcheval.metrics import FrechetInceptionDistance
from torchvision.utils import save_image 
from init import init
from omegaconf import DictConfig
from pathlib import Path
from sampler import sample
from utils import float2tensor


def create_image_directories() -> tuple[Path, Path]:
    """
    Create directories for saving images.
    """
    gen_path, ref_path = Path("./data/generated/"), Path("./data/reference")
    gen_path.mkdir(parents=True, exist_ok=True)
    ref_path.mkdir(parents=True, exist_ok=True)
    return gen_path, ref_path

@hydra.main(version_base=None, config_path="../config", config_name="config")
def fid(cfg: DictConfig):
    seed = torch.random.initial_seed()  # retrieve current seed

    # Initialization
    init_tuple = init(cfg)
    model, diffusion, info, dl = init_tuple.model, init_tuple.diffusion, init_tuple.info, init_tuple.dl
    gen_path, ref_path = create_image_directories()

    # Don't use the checkpoint seed for sampling
    torch.manual_seed(seed)

    # Sample and display
    try:
        sampling_method = cfg.common.sampling.method
    except:
        sampling_method = "euler"

    batch_size = 10
    N, C, H, W = batch_size, info.image_channels, info.image_size, info.image_size

    i = 0
    for (images, _) in dl.test:
        for image in images:
            save_image(float2tensor(image), ref_path / f"{i}.png")
            i +=1
        
    i = 0
    for _ in tqdm(range(0, 10_000, batch_size)):  # literature: on 50_000 generated images
        fake_samples = sample(
                            N, C, H, model, diffusion,
                            uncond_label=info.num_classes,
                            label=cfg.common.sampling.label,
                            cfg_scale=cfg.common.sampling.cfg_scale,
                            num_steps=cfg.common.sampling.num_steps,
                            track_inter=False,
                            sampling_method=sampling_method
                            )
        for image in fake_samples:
            save_image(float2tensor(image), gen_path / f"{i}.png")
            i +=1
    
if __name__ == "__main__":
    fid()
