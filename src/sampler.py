import hydra
import numpy as np
import torch

from diffusion import Diffusion
from init import init
from utils import display, save

from omegaconf import DictConfig
from typing import Optional, Callable


def expand(sigma: torch.Tensor, num: int) -> torch.Tensor:
    return sigma.expand(num)

def uncond_method(
        sigmas: torch.Tensor,
        X_noisy: torch.Tensor,
        D: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        uncond_label: Optional[int],
        stochastic: bool=False,
        euler_improved: bool=False,
    ) -> tuple[torch.Tensor, ...]:
    # N, S_min, S_max, S_churn, S_noise = sigmas.numel(), 0.05, 80, 36, 1  # 1.003
    S_noise = 1
    if uncond_label: uncond_label = torch.tensor(uncond_label, device=X_noisy.device).expand(X_noisy.shape[0])

    # Track the iterative procedure
    X_inter = torch.zeros(size=(len(sigmas)+1, ) + X_noisy.shape)
    X_inter[0] = X_noisy
    
    derivative = lambda x, sigma: (x - D(x, expand(sigma, x.shape[0]), uncond_label)) / sigma
    
    for i, sigma in enumerate(sigmas):
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        
        if stochastic:
            gamma = np.sqrt(2)-1  # min(S_churn/N, np.sqrt(2)-1) if (sigma >= S_min and sigma <= S_max) else 0   # np.sqrt(2)-1  TODO fix this
            sigma_new = sigma + gamma*sigma
            X_noisy = X_noisy + torch.sqrt(sigma_new**2-sigma**2)*torch.randn(X_noisy.shape, device=X_noisy.device)*S_noise  # TODO fix S_noise

            sigma = sigma_new

        d = derivative(X_noisy, sigma)
        X_next = X_noisy + (sigma_next - sigma)*d  # Perform one step of Euler's method

        if euler_improved and sigma_next != 0:
            d_next = derivative(X_next, sigma_next)
            X_next = X_noisy + (sigma_next - sigma)*(d + d_next)/2
        X_noisy = X_next 

        # Track the iterative procedure
        X_inter[i+1] = X_noisy
    return X_noisy, X_inter

def conditional_method(
        sigmas: torch.Tensor,
        X_noisy: torch.Tensor,
        D: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        label: int,
        uncond_label: int,
        cfg_scale: float=0,
        stochastic: bool=False,
        euler_improved: bool=False
    ) -> tuple[torch.Tensor, ...]:
    
    label = torch.tensor(label, device=X_noisy.device).expand(X_noisy.shape[0])
    uncond_label = torch.tensor(uncond_label, device=X_noisy.device).expand(X_noisy.shape[0])

    # Track the iterative procedure
    X_inter = torch.zeros(size=(len(sigmas)+1, ) + X_noisy.shape)
    X_inter[0] = X_noisy

    def derivative(x, sigma):
        s = expand(sigma, x.shape[0])
        # TODO: concat
        X_denoised_uncond = D(x, s, uncond_label)  # based on our model, try to denoise X_noisy
        X_denoised_cond   = D(x, s, label)
        
        # Classifier-Free Guidance
        score_uncond_estimate = (x - X_denoised_uncond)/sigma**2  # of x_noisy
        score_cond_estimate   = (x - X_denoised_cond)/sigma**2    # of x_noisy given label
        s_estimate = torch.lerp(score_uncond_estimate, score_cond_estimate, cfg_scale)
        
        return sigma*s_estimate  # derivative
    
    for i, sigma in enumerate(sigmas):
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        
        d = derivative(X_noisy, sigma)
        X_next = X_noisy + (sigma_next - sigma)*d  # Perform one step of Euler's method

        if euler_improved and sigma_next != 0:
            d_next = derivative(X_next, sigma_next)
            X_next = X_noisy + (sigma_next - sigma)*(d + d_next)/2
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
        track_inter: bool=False,
        sampling_method: str="euler"
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
        X_noisy, X_inter = uncond_method(sigmas, X_noisy, D, uncond_label,
                                         stochastic="stochastic" in sampling_method,
                                         euler_improved="heun" in sampling_method)
    else:
        X_noisy, X_inter = conditional_method(sigmas, X_noisy, D, label, uncond_label, cfg_scale,
                                              stochastic="stochastic" in sampling_method,
                                              euler_improved="heun" in sampling_method)
    model.train()

    if track_inter:
        return X_noisy, X_inter
    # Final X_noisy contains the sampled images
    return X_noisy

@hydra.main(version_base=None, config_path="../config", config_name="config")
def sampler(cfg: DictConfig):
    seed = torch.random.initial_seed()  # retrieve current seed

    # Initialization
    init_tuple = init(cfg)
    model, diffusion, info = init_tuple.model, init_tuple.diffusion, init_tuple.info

    # Don't use the checkpoint seed for sampling
    torch.manual_seed(seed)

    # Sample and display
    try:
        sampling_method = cfg.common.sampling.method
    except:
        sampling_method = "euler"
    N, C, H, W = 8*8, info.image_channels, info.image_size, info.image_size
    samples, samples_inter = sample(
            N, C, H, model, diffusion,
            uncond_label=info.num_classes,
            label=cfg.common.sampling.label,
            cfg_scale=cfg.common.sampling.cfg_scale,
            num_steps=cfg.common.sampling.num_steps,
            track_inter=True,
            sampling_method=sampling_method
        )
    dataset_name = str.lower(cfg.dataset.name)

    # display(samples)
    save(samples, f"./src/images/uncond_samples_{dataset_name}_16.png")
    # Display intermediate generation steps 
    # for the first generated picture
    # display(samples_inter[:, 0].view(-1, C, H, W))
    save(samples_inter[:, 0].view(-1, C, H, W), f"./src/images/iterative_denoising_process_{dataset_name}.png")
    

if __name__ == "__main__":
    sampler()