import hydra
import torch

from cleanfid import fid
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image 

from data import DataLoaders
from init import init_sampling
from sampler import sample
from utils import float2tensor


def create_directories(
        dataset_name: str,
        sampling_method: str,
        num_gen: int,
        gen_prefix: str=''
    ) -> tuple[list[Path], Path, Path]:
    """
    Create directories for saving images and fids.
    """
    dataset_name = str.lower(dataset_name)
    ref_paths = [Path(f"./data/fid/{dataset_name}/reference_{split}/") for split in DataLoaders._fields]
    gen_path = Path(f"./data/fid/{dataset_name}/{sampling_method}/{gen_prefix}generated_{num_gen//1000}k/")
    fid_path = Path(f"./results/fid/{dataset_name}/{sampling_method}/")
    for ref_path in ref_paths: ref_path.mkdir(parents=True, exist_ok=True)
    gen_path.mkdir(parents=True, exist_ok=True)
    fid_path.mkdir(parents=True, exist_ok=True)
    return ref_paths, gen_path, fid_path

@torch.inference_mode()
@hydra.main(version_base=None, config_path="../config", config_name="config")
def compute_fid(cfg: DictConfig):
    # Change this to True if you have a class-conditioned model
    conditional = False  # True
    
    model, diffusion, info, dl, sampling_method, _, cfgscale_str = init_sampling(cfg)
    batch_size, num_gen = 10, 50_000  # literature: on 50_000 generated images
    N, C, H, W = batch_size, info.image_channels, info.image_size, info.image_size

    gen_prefix = 'cond_' if conditional else ''
    ref_paths, gen_path, fid_path = create_directories(cfg.dataset.name, sampling_method, num_gen, gen_prefix)

    # Reference sets
    for ref_path, dataloader, split in zip(ref_paths, dl, DataLoaders._fields):
        if not(Path.exists(ref_path / "0.png")):
            print(f"Start saving images from {split} ref set")
            i = 0
            for (images, _) in tqdm(dataloader):
                for image in images:
                    save_image(float2tensor(image), ref_path / f"{i}.png")
                    i +=1
    
    # Generated set
    if not(Path.exists(gen_path / f"{num_gen-1}.png")):
        print(f"Start saving generated images")
        i = 0
        for _ in tqdm(range(0, num_gen, batch_size)):
            fake_samples = sample(
                N, C, H, model, diffusion,
                uncond_label=info.num_classes,
                label=torch.randint(0, info.num_classes, size=(N, )) if (conditional and (info.num_classes is not None)) else None,
                cfg_scale=cfg.common.sampling.cfg_scale,
                num_steps=cfg.common.sampling.num_steps,
                track_inter=False,
                sampling_method=sampling_method
            )
            for image in fake_samples:
                save_image(float2tensor(image), gen_path / f"{i}.png")
                i +=1
    
    for ref_path, split in zip(ref_paths, DataLoaders._fields):
        score = fid.compute_fid(str(gen_path), str(ref_path))  # TODO: Not optimal, it always recomputes gen statistics
        txt = f"FID {split} {num_gen//1000}k: {score}"
        print(txt)
        with open(str(fid_path / f"{gen_prefix}fid_{split}_{num_gen//1000}k.txt"), 'w') as f:
            print(txt, file=f)

if __name__ == "__main__":
    compute_fid()
