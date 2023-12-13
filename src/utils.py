import matplotlib.pyplot as plt
import os
import shutil
import torch
import wandb

from pathlib import Path
from PIL import Image
from torchvision.utils import make_grid

def copy_config(run: wandb.sdk.wandb_run.Run,
                begin_date: str,
                config_name: str="config") -> None:
    path = f'checkpoints/{begin_date}/run_{run.id}'
    os.makedirs(path, exist_ok=True)
    shutil.copyfile(f"config/{config_name}.yaml", path + "/config.yaml")

def copy_chkpt(run: wandb.sdk.wandb_run.Run,
               begin_date: str,
               chkpt_path: Path) -> None:
    path = f'checkpoints/{begin_date}/run_{run.id}'
    os.makedirs(path, exist_ok=True)
    shutil.copyfile(chkpt_path, path + "/checkpoint.pth")

def float2uint8(x: torch.Tensor) -> torch.Tensor:
    # Clamp/clip and convert to displayable format
    return x.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]

def float2tensor(x: torch.Tensor) -> torch.Tensor:
    # Clamp/clip and convert to format used for FID
    return x.clamp(-1, 1).add(1).div(2)  # [-1., 1.] -> [0., 1.]

def display(x: torch.Tensor,
            nrow: int=8,
            padding: int=2) -> None:
    x = make_grid(float2uint8(x), nrow=nrow, padding=padding)
    x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
    plt.imshow(x)
    plt.show()

def save(x: torch.Tensor,
         save_path: str,
         nrow: int=8,
         padding: int=2) -> None:
    x = make_grid(float2uint8(x), nrow=nrow, padding=padding)
    x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
    x.save(save_path)

def save_individually(x: torch.Tensor,
                      save_path: str) -> None:
    x.save(save_path)