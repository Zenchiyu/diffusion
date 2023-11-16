import os
import shutil
import wandb

from pathlib import Path


def copy_config(run: wandb.sdk.wandb_run.Run,
                begin_date: str) -> None:
    path = f'checkpoints/{begin_date}/run_{run.id}'
    os.makedirs(path, exist_ok=True)
    shutil.copyfile("config/config.yaml", path + "/config.yaml")

def copy_chkpt(run: wandb.sdk.wandb_run.Run,
               begin_date: str,
               chkpt_path: Path) -> None:
    path = f'checkpoints/{begin_date}/run_{run.id}'
    os.makedirs(path, exist_ok=True)
    shutil.copyfile(chkpt_path, path + "/checkpoint.pth")
