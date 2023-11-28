import hydra
import torch

from diffusion import Diffusion
from init import init
from utils import display

from omegaconf import DictConfig
from typing import Optional, Callable


def euler_method(
        sigmas: torch.Tensor,
        X_noisy: torch.Tensor,
        D: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        uncond_label: Optional[int],
        improved: bool=False
    ) -> tuple[torch.Tensor, ...]:

    if uncond_label: uncond_label = torch.tensor(uncond_label, device=X_noisy.device).expand(X_noisy.shape[0])

    # Track the iterative procedure
    X_inter = torch.zeros(size=(len(sigmas)+1, ) + X_noisy.shape)
    X_inter[0] = X_noisy
    
    for i, sigma in enumerate(sigmas):
        s = sigma.expand(X_noisy.shape[0])
        X_denoised_uncond = D(X_noisy, s, uncond_label)  # based on our model, try to denoise X_noisy
        
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (X_noisy - X_denoised_uncond) / sigma     # derivative
        X_noisy = X_noisy + d * (sigma_next - sigma)  # Perform one step of Euler's method

        # Track the iterative procedure
        X_inter[i+1] = X_noisy
    return X_noisy, X_inter

def euler_method_conditional(
        sigmas: torch.Tensor,
        X_noisy: torch.Tensor,
        D: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        label: int,
        uncond_label: int,
        cfg_scale: float=0,
        improved: bool=True
    ) -> tuple[torch.Tensor, ...]:
    
    label = torch.tensor(label, device=X_noisy.device).expand(X_noisy.shape[0])
    uncond_label = torch.tensor(uncond_label, device=X_noisy.device).expand(X_noisy.shape[0])

    # Track the iterative procedure
    X_inter = torch.zeros(size=(len(sigmas)+1, ) + X_noisy.shape)
    X_inter[0] = X_noisy
    
    for i, sigma in enumerate(sigmas):
        x_noisy = X_noisy.repeat((2,)+(1,)*(X_noisy.ndim-1))
        s = sigma.expand(2*X_noisy.shape[0])
        l = torch.cat([uncond_label, label])

        x_denoised = D(x_noisy, s, l)

        # Classifier-Free Guidance
        score_uncond_estimate, score_cond_estimate = ((x_noisy - x_denoised)/sigma**2).chunk(2, dim=0)
        s_estimate = torch.lerp(score_uncond_estimate, score_cond_estimate, cfg_scale)
        
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = sigma*s_estimate  # derivative
        X_next = X_noisy + d * (sigma_next - sigma)  # Perform one step of Euler's method
        
        if improved and sigma_next != 0:
            # Heun a.k.a. improved Euler method:
            x_noisy = X_next.repeat((2,)+(1,)*(X_next.ndim-1))
            s = sigma_next.expand(2*X_next.shape[0])

            x_denoised = D(x_noisy, s, l)
            # CFG
            score_uncond_estimate, score_cond_estimate = ((x_noisy - x_denoised)/sigma_next**2).chunk(2, dim=0)
            s_estimate_next = torch.lerp(score_uncond_estimate, score_cond_estimate, cfg_scale)
            
            d_next = sigma_next*s_estimate_next
            X_next = X_noisy + (d/2 + d_next/2) * (sigma_next - sigma)

        X_noisy = X_next

        # Track the iterative procedure
        X_inter[i+1] = X_noisy
    return X_noisy, X_inter

@torch.no_grad()
def sample(
        num_samples: int,
        image_channels: int,
        image_size: int,
        model: torch.nn.Module,
        diffusion: Diffusion,
        uncond_label: Optional[int],
        label: Optional[int]=None,
        cfg_scale: float=0,
        num_steps: int=50,
        track_inter: bool=False
    ) -> torch.Tensor:

    model.eval()
    sigmas = diffusion.build_sigma_schedule(steps=num_steps, rho=7)  # Sequence of decreasing sigmas
    cin, cout, cskip, cnoise = diffusion.cin, diffusion.cout, diffusion.cskip, diffusion.cnoise
    
    # Denoiser
    D = lambda X_noisy, sigma, label: cskip(sigma)*X_noisy+cout(sigma)*model(cin(sigma)*X_noisy, cnoise(sigma), label)

    # Initialize with pure gaussian noise ~ N(0, sigmas[0])
    # Initial condition of the differential equation
    X_noisy = torch.randn(num_samples, image_channels,
                        image_size, image_size,
                        device=diffusion.device) * sigmas[0]
    
    if label is None or uncond_label is None:
        X_noisy, X_inter = euler_method(sigmas, X_noisy, D, uncond_label)
    else:
        X_noisy, X_inter = euler_method_conditional(sigmas, X_noisy, D, label, uncond_label, cfg_scale)
    model.train()

    if track_inter:
        return X_noisy, X_inter
    # Final X_noisy contains the sampled images
    return X_noisy

@hydra.main(version_base=None, config_path="../config", config_name="config")
def sampler(cfg: DictConfig):
    # Initialization
    init_tuple = init(cfg)
    model, diffusion, info = init_tuple.model, init_tuple.diffusion, init_tuple.info

    # Don't use the checkpoint seed for sampling
    torch.seed()

    # Sample and display
    N, C, H, W = 8*8, info.image_channels, info.image_size, info.image_size
    samples, samples_inter = sample(
            N, C, H, model, diffusion,
            uncond_label=info.num_classes,
            label=cfg.common.sampling.label,
            cfg_scale=cfg.common.sampling.cfg_scale,
            num_steps=cfg.common.sampling.num_steps,
            track_inter=True
        )

    display(samples)
    # Display intermediate generation steps 
    # for the first generated picture
    display(samples_inter[:, 0].view(-1, C, H, W))

if __name__ == "__main__":
    sampler()