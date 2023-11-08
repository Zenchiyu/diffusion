import torch
from init import init
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def trainer(cfg: DictConfig):
    if cfg.wandb.mode == "online":
        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                         **cfg.wandb)
    
    model, optimizer, criterion, diffusion, dl, info, device = init(cfg)

    # Training
    for e in tqdm(range(cfg.common.nb_epochs)):
        for X, y in dl.train:
            X = X.to(device=device)

            noise_level = diffusion.sample_sigma(X.shape[0])
            cin = diffusion.cin(noise_level)
            cout = diffusion.cout(noise_level)
            cskip = diffusion.cskip(noise_level)
            cnoise = diffusion.cnoise(noise_level)
            
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