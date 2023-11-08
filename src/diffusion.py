# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import torch


class Diffusion:
    def __init__(self,
                 sigma_data: float,
                 sigma_min: float=2e-3,
                 sigma_max: float=80) -> None:
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @staticmethod
    def add_noise(x: torch.FloatTensor,
                  noise_level: float):
        # noising procedure using re-param trick
        return x + noise_level*torch.normal(0, 1, size=x.shape)

    def sample_sigma(self,
                     n: list[int]|tuple[int]|int,
                     loc: float=-1.2,
                     scale: float=1.2):
        # From Eloi's template
        # exp(Z) where Z follows a normal distribution then clip
        return (torch.randn(n) * scale + loc).exp().clip(self.sigma_min, self.sigma_max)

    def build_sigma_schedule(self, steps, rho=7):
        # From Eloi's template
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas
    
if __name__ == "__main__":
    pass