import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from diffusion import Diffusion
from data import load_dataset_and_make_dataloaders  # Made by Eloi
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def trainer(cfg: DictConfig):
    if cfg.wandb.mode == "online":
        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                         **cfg.wandb)
    gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if gpu else 'cpu')
    
    # DataLoaders
    dl, info = load_dataset_and_make_dataloaders(
        dataset_name=cfg.dataset.dataset_name,  # can be "FashionMNIST" for instance
        root_dir=cfg.dataset.root_dir, # choose the directory to store the data 
        batch_size=cfg.dataset.batch_size,  # 32
        num_workers=cfg.dataset.num_workers,   # can use more workers if see the GPU is waiting for the batches
        pin_memory=gpu,  # use pin memory if plan to move the data to GPU
    )
    # TODO: use the same dataloaders at evaluation time..
    
    diffusion = Diffusion(info.sigma_data,
                          cfg.diffusion.sigma_min,
                          cfg.diffusion.sigma_max)

    # XXX: info.sigma_data  # estimation of the std based on a "huge" batch
    # Model and criterion
    model = Model(info.image_channels,
                  info.nb_channels,
                  cfg.model.num_blocks,
                  cfg.model.cond_channels)
    criterion = nn.MSELoss()
    model.to(device=device)
    criterion.to(device=device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr)

    # Training
    for e in tqdm(range(cfg.common.nb_epochs)):
        for X, y in dl.train:
            cin = diffusion.cin(noise_level)
            cout = diffusion.cout(noise_level)
            cskip = diffusion.cskip(noise_level)
            cnoise = diffusion.cnoise(noise_level)
            
            noise_level = 0
            X_noisy = diffusion.add_noise(X, noise_level)
            output = model(cin*X_noisy, cnoise)
            target = (X-cskip*X_noisy)/cout

            loss = criterion(output, target)  # MSE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # TODO: wandb log

if __name__ == "__main__":
    pass