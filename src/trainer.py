# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import torch
from init import init
from tqdm import tqdm
from sampler import sample, save

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def trainer(cfg: DictConfig):
    if cfg.wandb.mode == "online":
        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                         **cfg.wandb)
    
    model, optimizer, criterion, diffusion, dl, info, device = init(cfg)

    # TODO: add something to restart a run

    # Training
    for e in tqdm(range(cfg.common.nb_epochs)):
        acc_loss = 0
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

            acc_loss += loss.item()

        samples = sample(32, model, diffusion)
        save(samples, cfg.sampling.save_path)
        # TODO: wandb log
        if cfg.wandb.mode == "online":
            wandb.log({"epoch": e,
                       "acc_loss": acc_loss,
                       })
    
    if cfg.wandb.mode == "online":
        wandb.finish()

if __name__ == "__main__":
    trainer()