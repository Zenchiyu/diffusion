# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from diffusion import Diffusion
from data import load_dataset_and_make_dataloaders  # Made by Eloi
from pathlib import Path


def init(cfg):
    gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if gpu else 'cpu')
    
    # DataLoaders
    dl, info = load_dataset_and_make_dataloaders(
        dataset_name=cfg.dataset.name,  # can be "FashionMNIST" for instance
        root_dir=cfg.dataset.root_dir, # choose the directory to store the data 
        batch_size=cfg.dataset.batch_size,  # 32
        num_workers=cfg.dataset.num_workers,   # can use more workers if see the GPU is waiting for the batches
        pin_memory=gpu,  # use pin memory if plan to move the data to GPU
    )
    # TODO: use the same dataloaders at evaluation time..

    # Create directory to save pictures of our samples
    save_path = Path(cfg.common.sampling.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Create directory to save checkpoints
    chkpt_path = Path(cfg.common.training.chkpt_path)
    chkpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    diffusion = Diffusion(info.sigma_data,
                          cfg.diffusion.sigma_min,
                          cfg.diffusion.sigma_max)
    # XXX: info.sigma_data is an estimation of the std based on a "huge" batch

    # Model and criterion
    model = Model(info.image_channels,
                  cfg.model.nb_channels,
                  cfg.model.num_blocks,
                  cfg.model.cond_channels)
    criterion = nn.MSELoss()
    model.to(device=device)
    criterion.to(device=device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr)
    return model, optimizer, criterion, diffusion, dl, info, device, save_path, chkpt_path