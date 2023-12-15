import hydra
import torch

from tqdm import tqdm
from torchvision.utils import save_image 
from init import init
from omegaconf import DictConfig
from pathlib import Path
from utils import float2tensor


def create_image_directory(
        dataset_name: str,
        sampling_method: str
    ) -> tuple[Path, Path]:
    """
    Create directory for saving nearest neigbhors images.
    """
    dataset_name = str.lower(dataset_name)
    # Pick pictures generated from fid.py 10k (assume already did)
    gen_path = Path(f"./data/fid/{dataset_name}/{sampling_method}/generated_10k/")
    nn_path = Path(f"./results/images/{dataset_name}/{sampling_method}/nearest_neighbors/")
    nn_path.mkdir(parents=True, exist_ok=True)
    return gen_path, nn_path

@hydra.main(version_base=None, config_path="../config", config_name="config")
def nearest_neighbors(cfg: DictConfig):
    

if __name__ == "__main__":
    nearest_neighbors()