import hydra
import torch

from torchvision.io import read_image
from init import init_sampling
from omegaconf import DictConfig
from pathlib import Path
from utils import save


def create_image_directory(
        dataset_name: str,
        sampling_method: str
    ) -> tuple[Path, Path, Path]:
    """
    Create directory for saving nearest neigbhors images.
    """
    dataset_name = str.lower(dataset_name)
    # Pick 50k already [conditionally] generated pictures and training set pictures saved using fid.py
    ref_path = Path(f"./data/fid/{dataset_name}/reference_train/")
    gen_path = Path(f"./data/fid/{dataset_name}/{sampling_method}/cond_generated_50k/")
    nn_path = Path(f"./results/images/{dataset_name}/{sampling_method}/nearest_neighbors/")
    nn_path.mkdir(parents=True, exist_ok=True)
    return ref_path, gen_path, nn_path

@torch.inference_mode()
@hydra.main(version_base=None, config_path="../config", config_name="config")
def nearest_neighbors(cfg: DictConfig):
    init_tuple = init_sampling(cfg)
    info, dl, sampling_method = init_tuple.info, init_tuple.dl, init_tuple.sampling_method
    ref_path, gen_path, nn_path = create_image_directory(cfg.dataset.name, sampling_method)
    
    num_gen, num_nn, num_ref = 9, 20, dl.train.sampler.num_samples
    print(f"{num_nn} Nearest neighbors measured by the L2 distance between {num_gen} generated and {num_ref} train images.")
    
    x = torch.stack([(read_image(str(gen_path / f"{i}.png"))/255)*2-1 for i in range(num_gen)], dim=0).view(num_gen, -1)
    y = torch.stack([(read_image(str(ref_path / f"{i}.png"))/255)*2-1 for i in range(num_ref)], dim=0).view(-1, num_ref)
    dist = torch.linalg.norm(x, dim=1, keepdim=True)**2 + torch.linalg.norm(y, dim=0, keepdim=True)**2
    dist -= 2*(x @ y)

    indices = torch.topk(dist, k=num_nn, dim=1, largest=False).indices
    nn = torch.cat([x.unsqueeze(1), y.view(num_ref, -1)[indices]], dim=1)
    nn = nn.view(num_gen*(1+num_nn), 3, info.image_size, -1)  # 3 channels even if grayscale
    save(nn, nn_path / "image_space_nn.png", nrow=num_nn+1)

    # TODO: fix num_ref

if __name__ == "__main__":
    nearest_neighbors()