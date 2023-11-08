# Made by Stephane Nguyen following
# notebooks/instructions.ipynb
import torch

class Diffusion():
    def __init__(self) -> None:
        pass

    @staticmethod
    def add_noise(x: torch.FloatTensor,
                  noise_level: float):
        # noising procedure using re-param trick
        return x + noise_level*torch.normal(0, 1, size=x.shape)


if __name__ == "__main__":
    pass