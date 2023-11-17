# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import hydra
import torch
import wandb

from init import init
from sampler import sample
from utils import copy_config, copy_chkpt, save

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def trainer(cfg: DictConfig):
    # Initialization
    init_tuple = init(cfg)
    model, optimizer, criterion, diffusion = init_tuple.model, init_tuple.optimizer, init_tuple.criterion, init_tuple.diffusion
    dl, info, device = init_tuple.dl, init_tuple.info, init_tuple.device,
    save_path, chkpt_path = init_tuple.save_path, init_tuple.chkpt_path
    nb_epochs_finished = init_tuple.nb_epochs_finished
    
    begin_date = init_tuple.begin_date
    seed = torch.random.initial_seed()  # retrieve current seed

    if cfg.wandb.mode == "online":
        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                         **cfg.wandb)
        run.watch(model, criterion, log="all", log_graph=True)
        copy_config(run, begin_date=begin_date)

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

            # XXX: Classifier-Free Guidance
            if torch.rand() < cfg.common.training.p_uncond:
                clabel = None
            else:
                clabel = y/info.num_classes-0.5  # [-0.5, 0.5] more or less                

            output = model(cin*X_noisy, cnoise, clabel)
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
                    "acc_losses": acc_losses,
                    "seed": seed,
                    "begin_date": begin_date},
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