import hydra
import torch

from cleanfid import fid
from tqdm import tqdm
from torchvision.utils import save_image 
from init import init
from omegaconf import DictConfig
from pathlib import Path
from sampler import sample
from utils import float2tensor


def create_directories(
        dataset_name: str,
        sampling_method: str
    ) -> tuple[Path, Path, Path]:
    """
    Create directories for saving images and fid.
    """
    dataset_name = str.lower(dataset_name)
    ref_path = Path(f"./data/fid/{dataset_name}/reference/")
    gen_path = Path(f"./data/fid/{dataset_name}/{sampling_method}/generated/")
    fid_path = Path(f"./results/fid/{dataset_name}/{sampling_method}/")
    ref_path.mkdir(parents=True, exist_ok=True)
    gen_path.mkdir(parents=True, exist_ok=True)
    fid_path.mkdir(parents=True, exist_ok=True)
    return ref_path, gen_path, fid_path

@hydra.main(version_base=None, config_path="../config", config_name="config")
def compute_fid(cfg: DictConfig):
    seed = torch.random.initial_seed()  # retrieve current seed

    # Initialization
    init_tuple = init(cfg)
    model, diffusion, info, dl = init_tuple.model, init_tuple.diffusion, init_tuple.info, init_tuple.dl
    try:
        sampling_method = cfg.common.sampling.method
    except:
        sampling_method = "euler"
    ref_path, gen_path, fid_path = create_directories(cfg.dataset.name, sampling_method)

    # Don't use the checkpoint seed for sampling
    torch.manual_seed(seed)

    # Sample and display
    batch_size = 10
    num_gen = 10_000  # literature: on 50_000 generated images
    N, C, H, W = batch_size, info.image_channels, info.image_size, info.image_size

    # Don't save again the images from the test set.
    if not(Path.exists(ref_path / "0.png")):
        i = 0
        for (images, _) in dl.test:
            for image in images:
                save_image(float2tensor(image), ref_path / f"{i}.png")
                i +=1
    
    # Don't generate & save if already generated & saved
    if not(Path.exists(gen_path / f"{num_gen-1}.png")):
        i = 0
        for _ in tqdm(range(0, num_gen, batch_size)):
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
    
    score = fid.compute_fid(str(gen_path), str(ref_path))
    print(f"FID: {score}")
    with open(str(fid_path / "fid.txt"), 'w') as f:
        print(f"FID: {score}", file=f)


if __name__ == "__main__":
    compute_fid()
