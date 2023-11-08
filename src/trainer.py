# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import torch
from init import init
from sampler import sample, save
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def trainer(cfg: DictConfig):
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.wandb.mode == "online":
        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                         **cfg.wandb)
    
    model, optimizer, criterion, diffusion, dl, info, device, save_path = init(cfg)
    print(f"\n\nDataset: {cfg.dataset.name}, Using device: {device}")
    
    # TODO: add something to restart a run

    # Training
    for e in tqdm(range(cfg.common.nb_epochs)):
        acc_loss = 0
        for X, y in dl.train:
            X = X.to(device=device)  # N x C x H x W

            noise_level = diffusion.sample_sigma(X.shape[0])
            cin = diffusion.cin(noise_level)
            cout = diffusion.cout(noise_level)
            cskip = diffusion.cskip(noise_level)
            cnoise = diffusion.cnoise(noise_level)
            
            X_noisy = diffusion.add_noise(X, noise_level.view(-1, 1, 1, 1))
            output = model(cin*X_noisy, cnoise)
            target = (X-cskip*X_noisy)/cout

            loss = criterion(output, target)  # MSE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_loss += loss.item()

        samples = sample(32, model, diffusion)  # it's switching between eval and train modes
        save(samples)
        # TODO: wandb log
        if cfg.wandb.mode == "online":
            wandb.log({"epoch": e,
                       "acc_loss": acc_loss,
                       "samples": wandb.Image(save_path)})
    
    if cfg.wandb.mode == "online":
        wandb.finish()

if __name__ == "__main__":
    trainer()