# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import hydra
import matplotlib.pyplot as plt
import torch

from diffusion import Diffusion
from init import init

from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid


def sample(num_samples: int,
           image_channels: int,
           image_size: int,
           model: torch.nn.Module,
           diffusion: Diffusion,
           num_steps: int=50,
           track_inter: bool=False) -> torch.Tensor:
    model.eval()
    sigmas = diffusion.build_sigma_schedule(steps=num_steps, rho=7)  # Sequence of decreasing sigmas
    cin = diffusion.cin
    cout = diffusion.cout
    cskip = diffusion.cskip
    cnoise = diffusion.cnoise
    D = lambda X_noisy, sigma: cskip(sigma)*X_noisy+cout(sigma)*model(cin(sigma)*X_noisy, cnoise(sigma))

    X_noisy = torch.randn(num_samples, image_channels,
                          image_size, image_size,
                          device=diffusion.device) * sigmas[0]  # Initialize with pure gaussian noise ~ N(0, sigmas[0])
    
    # Track the iterative procedure
    X_inter = torch.zeros(size=(len(sigmas)+1, ) + X_noisy.shape)
    X_inter[0] = X_noisy

    for i, sigma in enumerate(sigmas):
        with torch.no_grad():
            X_denoised = D(X_noisy, sigma)  # based on our model, try to denoise X_noisy
        
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (X_noisy - X_denoised) / sigma  # derivative
        X_noisy = X_noisy + d * (sigma_next - sigma)  # Perform one step of Euler's method

        # Track the iterative procedure
        X_inter[i+1] = X_noisy
    
    model.train()
    if track_inter:
        return X_noisy, X_inter
    # Final X_noisy contains the sampled images
    return X_noisy

def float2uint8(x: torch.Tensor) -> torch.Tensor:
    # Clamp/clip and convert to displayable format
    return x.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]

def display(x: torch.Tensor,
            nrow: int=8) -> None:
    x = make_grid(float2uint8(x))
    x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
    plt.imshow(x)
    plt.show()

def save(x: torch.Tensor,
         save_path: str,
         nrow: int=8) -> None:
    x = make_grid(float2uint8(x))
    x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
    x.save(save_path)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def sampler(cfg: DictConfig):
    model, _, _, diffusion, dl, info, device, save_path, chkpt_path = init(cfg)

    # Load model state dict from checkpoint:
    chkpt = torch.load(chkpt_path)
    model.load_state_dict(chkpt["model_state_dict"])

    # Sample and display
    num_samples = 8*8
    samples, samples_inter = sample(num_samples,
                                    info.image_channels,
                                    info.image_size,
                                    model,
                                    diffusion,
                                    num_steps=cfg.common.sampling.num_steps,
                                    track_inter=True)
    display(samples)
    # Display for the first generated picture
    display(samples_inter[:, 0].view(-1, info.image_channels, 32, 32))

if __name__ == "__main__":
    sampler()