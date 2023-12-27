import hydra
import numpy as np
import torch

from diffusion import Diffusion
from init import init_sampling
from utils import save, expand

from omegaconf import DictConfig
from typing import Optional, Callable


def sampling_process(
        X_noisy: torch.Tensor,
        sigmas: torch.Tensor,
        derivative: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        stochastic: bool=False,
        euler_improved: bool=False,
    ) -> tuple[torch.Tensor, ...]:
    # N, S_min, S_max, S_churn, S_noise = sigmas.numel(), 0.05, 80, 36, 1  # 1.003
    S_noise = 1

    # Track the iterative procedure
    X_inter = torch.zeros(size=(len(sigmas)+1, ) + X_noisy.shape)
    X_inter[0] = X_noisy
    
    for i, sigma in enumerate(sigmas):
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        
        if stochastic:
            gamma = np.sqrt(2)-1  # min(S_churn/N, np.sqrt(2)-1) if (sigma >= S_min and sigma <= S_max) else 0   # np.sqrt(2)-1  TODO fix this
            sigma_new = sigma + gamma*sigma
            X_noisy = X_noisy + torch.sqrt(sigma_new**2-sigma**2)*torch.randn(X_noisy.shape, device=X_noisy.device)*S_noise

            sigma = sigma_new

        d = derivative(X_noisy, sigma)
        X_next = X_noisy + (sigma_next - sigma)*d  # Perform one step of Euler's method

        if euler_improved and sigma_next != 0:
            d_next = derivative(X_next, sigma_next)
            X_next = X_noisy + (sigma_next - sigma)*(d + d_next)/2
        X_noisy = X_next 

        # Track the iterative procedure
        X_inter[i+1] = X_noisy
    return X_noisy, X_inter.transpose(0, 1)

def sample_chunked(nb_chunks=2, **kwargs) -> torch.Tensor:
    assert kwargs["num_samples"] % nb_chunks == 0,\
        "num_samples should be a multiple of the number of chunks"
    num_samples, label = kwargs["num_samples"]//nb_chunks, kwargs.get("label", None)
    chunked_labels = nb_chunks*[label] if ((label is None) or isinstance(label, int)) else label.chunk(nb_chunks)
    
    def new_kwargs(label: Optional[torch.Tensor]=None) -> torch.Tensor:
        return kwargs | {"num_samples": num_samples, "label": label}
    
    if kwargs.get("track_inter", False):
        return tuple(map(lambda el: torch.cat(el, dim=0), zip(*[sample(**new_kwargs(label)) for label in chunked_labels])))
    return torch.cat([sample(**new_kwargs(label)) for label in chunked_labels], dim=0)

@torch.inference_mode()
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
    if label is not None: label = torch.tensor(label, device=X_noisy.device).expand(X_noisy.shape[0])
    if uncond_label is not None: uncond_label = torch.tensor(uncond_label, device=X_noisy.device).expand(X_noisy.shape[0])

    if label is None or uncond_label is None:
        derivative = lambda x, sigma: (x - D(x, expand(sigma, x.shape[0]), uncond_label)) / sigma
    else:
        def derivative(x, sigma):
            s = expand(sigma, x.shape[0])
            # TODO: concat but be careful about batch norm
            X_denoised_uncond = D(x, s, uncond_label)  # based on our model, try to denoise X_noisy
            X_denoised_cond   = D(x, s, label)
            
            # Classifier-Free Guidance
            score_uncond_estimate = (x - X_denoised_uncond)/sigma**2  # of x_noisy
            score_cond_estimate   = (x - X_denoised_cond)/sigma**2    # of x_noisy given label
            s_estimate = torch.lerp(score_uncond_estimate, score_cond_estimate, cfg_scale)

            return sigma*s_estimate  # derivative
    
    X_noisy, X_inter = sampling_process(X_noisy, sigmas, derivative,
                                        stochastic="stochastic" in sampling_method,
                                        euler_improved="heun" in sampling_method)
    model.train()

    if track_inter:
        return X_noisy, X_inter
    # Final X_noisy contains the sampled images
    return X_noisy

@torch.inference_mode()
@hydra.main(version_base=None, config_path="../config", config_name="config")
def sampler(cfg: DictConfig):
    model, diffusion, info, _, sampling_method, path, cfgscale_str = init_sampling(cfg)

    # Sample and display
    prefix = "cond" if cfg.common.sampling.label else "uncond"
    suffix = f'_class_{cfg.common.sampling.label}_cfgscale_{cfgscale_str}' if cfg.common.sampling.cfg_scale else ''
    N, C, H, W = 8*8, info.image_channels, info.image_size, info.image_size
    kwargs = {
        "num_samples": N,
        "image_channels": C,
        "image_size": H,
        "model": model,
        "diffusion": diffusion,
        "uncond_label": info.num_classes,
        "label": cfg.common.sampling.label,
        "cfg_scale": cfg.common.sampling.cfg_scale,
        "num_steps": cfg.common.sampling.num_steps,
        "track_inter": True,
        "sampling_method": sampling_method,
    }
    samples, samples_inter = sample(**kwargs)   # sample_chunked(nb_chunks=8, **kwargs) if memory issues

    save(samples, path / f"{prefix}_{N}{suffix}.png")
    # Save intermediate generation steps for the first generated picture
    save(samples_inter[0, :].view(-1, C, H, W), path / f"iterative_denoising_process{suffix}.png")

if __name__ == "__main__":
    sampler()