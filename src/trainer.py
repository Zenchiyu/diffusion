# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import hydra
import torch
import wandb

from init import init
from sampler import sample, save
from utils import copy_config, copy_chkpt

from datetime import date
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def trainer(cfg: DictConfig):
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    
    # TODO: work with same dataset if continue a run!
    model, optimizer, criterion, diffusion, dl, info, device, save_path, chkpt_path = init(cfg)
    print(f"\n\nDataset: {cfg.dataset.name}, Using device: {device}")
    
    begin_date = str(date.today())

    if cfg.wandb.mode == "online":
        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                         **cfg.wandb)
        run.watch(model, criterion,
                           log="all", log_graph=True)
        
        copy_config(run, begin_date=begin_date)
        
    # Restart a run
    # https://fleuret.org/dlc/materials/dlc-handout-11-4-persistence.pdf
    nb_epochs_finished = 0
    try:
        # Load model state dict from checkpoint:
        chkpt = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(chkpt["model_state_dict"])
        optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        nb_epochs_finished = chkpt["nb_epochs_finished"]
        print(f"\nStarting from checkpoint with {nb_epochs_finished} finished epochs.")
    except FileNotFoundError:
        print("Starting from scratch.")

    # Training
    acc_losses = []
    for e in tqdm(range(nb_epochs_finished, cfg.common.nb_epochs)):
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

        acc_losses.append(acc_loss)
        samples = sample(8,
                         info.image_channels,
                         info.image_size,
                         model,
                         diffusion,
                         cfg.common.sampling.num_steps)  # it's switching between eval and train modes
        save(samples, str(save_path))
        # Save checkpoint
        torch.save({"nb_epochs_finished": e+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "acc_losses": acc_losses},
                    chkpt_path)
        
        if cfg.wandb.mode == "online":
            wandb.log({"epoch": e,
                       "acc_loss": acc_loss,
                       "samples": wandb.Image(str(save_path))})
            copy_chkpt(run, begin_date, chkpt_path)
        
    
    if cfg.wandb.mode == "online":
        wandb.finish()

if __name__ == "__main__":
    trainer()