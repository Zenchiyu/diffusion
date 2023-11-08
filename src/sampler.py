# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import matplotlib.pyplot as plt
import os
import torch
from diffusion import Diffusion
from init import init
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

def sample(num_samples: int,
           model: torch.nn.Module,
           diffusion: Diffusion) -> torch.tensor:
    model.eval()
    sigmas = diffusion.build_sigma_schedule(steps=50, rho=7)  # Sequence of decreasing sigmas
    cin = diffusion.cin
    cout = diffusion.cout
    cskip = diffusion.cskip
    cnoise = diffusion.cnoise
    D = lambda X_noisy, sigma: cskip(sigma)*X_noisy+cout(sigma)*model(cin(sigma)*X_noisy, cnoise(sigma))

    X_noisy = torch.randn(num_samples, 1, 32, 32) * sigmas[0]  # Initialize with pure gaussian noise ~ N(0, sigmas[0])
    for i, sigma in enumerate(sigmas):
        with torch.no_grad():
            X_denoised = D(X_noisy, sigma)  # based on our model, try to denoise X_noisy
        
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (X_noisy - X_denoised) / sigma
        
        X_noisy = X_noisy + d * (sigma_next - sigma)  # Perform one step of Euler's method
    model.train()
    # Final X_noisy contains the sampled images
    return X_noisy

def display(x: torch.tensor) -> None:
    # Clamp/clip and convert to displayable format
    x = x.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
    x = make_grid(x)
    x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
    plt.imshow(x)

def save(x: torch.tensor, save_path: str) -> None:
    # Clamp/clip and convert to displayable format
    x = x.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
    x = make_grid(x)
    x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
    x.save(save_path)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def sampler(cfg: DictConfig):
    if cfg.wandb.mode == "online":
        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                         **cfg.wandb)
    
    model, _, _, diffusion, dl, _, device, save_path, chkpt_path = init(cfg)

    # Load model state dict from checkpoint:
    chkpt = torch.load(chkpt_path)
    model.load_state_dict(chkpt["model_state_dict"])

    # Sample and display
    num_samples = 8
    samples = sample(num_samples, model, diffusion)
    display(samples)

if __name__ == "__main__":
    sampler()