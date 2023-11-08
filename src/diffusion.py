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

        self.cin = lambda sigma: (1/torch.sqrt(sigma_data**2 + sigma**2)).view(-1, 1, 1, 1)
        self.cout = lambda sigma: (sigma*sigma_data/torch.sqrt(sigma_data**2 + sigma**2)).view(-1, 1, 1, 1)
        self.cskip = lambda sigma: (sigma_data**2/(sigma_data**2 + sigma**2)).view(-1, 1, 1, 1)
        self.cnoise = lambda sigma: torch.log(sigma)/4

    @staticmethod
    def add_noise(x: torch.FloatTensor,
                  noise_level: float|torch.FloatTensor):
        # noising procedure using re-param trick
        return x + noise_level*torch.normal(0, 1, size=noise_level.shape)

    def sample_sigma(self,
                     n: list[int]|tuple[int]|int,
                     loc: float=-1.2,
                     scale: float=1.2):
        """
        Sample noise levels following a log-normal distribution.
        These noise levels will be used in the training phase,
        to create noisy samples by using "add_noise".
        """
        # From Eloi's template
        # exp(Z) where Z follows a normal distribution then clip
        return (torch.randn(n) * scale + loc).exp().clip(self.sigma_min, self.sigma_max)

    def build_sigma_schedule(self, steps, rho=7):
        """
        Build sigma schedule of decreasing noise levels
        that will be used in the sampling procedure, not training phase.
        """
        # From Eloi's template
        # TODO: find whether rho is like some temperature?
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas
    
if __name__ == "__main__":
    pass