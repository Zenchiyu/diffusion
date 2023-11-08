# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import torch

def sample():
    sigmas = build_sigma_schedule(steps=50, rho=7)  # Sequence of decreasing sigmas

    x = torch.randn(8, 1, 32, 32) * sigmas[0]  # Initialize with pure gaussian noise ~ N(0, sigmas[0])

    for i, sigma in enumerate(sigmas):
        
        with torch.no_grad():
            x_denoised = D(x, sigma)  
            # Where D(x, sigma) = cskip(sigma) * x + cout(sigma) * F(cin(sigma) * x, cnoise(sigma)) 
            # and F(.,.) is your neural network
        
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (x - x_denoised) / sigma
        
        x = x + d * (sigma_next - sigma)  # Perform one step of Euler's method

    # -> Final `x` contains the sampled images (8 here)

if __name__ == "__main__":
    pass