import torch
import torch.nn as nn
import torch.optim as optim

from models.unet import UNet
from diffusion import Diffusion
from data import load_dataset_and_make_dataloaders

from collections import namedtuple
from datetime import date
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Any


Init = namedtuple('Init', 'model optimizer criterion '+\
                  'diffusion dl info device nb_epochs_finished '+\
                  'begin_date save_path chkpt_path')
InitSample = namedtuple('InitSample', 'model diffusion info dl '+\
                        'sampling_method path cfgscale_str')

def create_save_directories(cfg: DictConfig) -> tuple[Path, Path]:
    """
    Create directories for saving samples and checkpoints.
    """
    save_path, chkpt_path = Path(cfg.common.sampling.save_path), Path(cfg.common.training.chkpt_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    chkpt_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path, chkpt_path

def load_chkpt(chkpt_path: Path, device: str|torch.device) -> tuple[Any, int, str]:
    """
    Load checkpoint if exists, set random seed, and handle run resuming.
    
    Note: seed is used to get the same training, validation sets splits
    when resuming our runs.

    Credits: https://fleuret.org/dlc/materials/dlc-handout-11-4-persistence.pdf
    """
    chkpt, nb_epochs_finished, begin_date, seed = None, 0, str(date.today()), torch.initial_seed()  # by default: random seed
    try:
        chkpt = torch.load(chkpt_path, map_location=device)
        nb_epochs_finished = chkpt.get("nb_epochs_finished", nb_epochs_finished)
        begin_date = chkpt.get("begin_date", begin_date)
        seed = chkpt.get("seed", seed)
        torch.manual_seed(seed)
        print(f"\nStarting from checkpoint with {nb_epochs_finished} finished epochs"+\
              f", and initial seed {seed} (=> same datasets).")
    except FileNotFoundError:
        print(f"Starting from scratch with random initial seed {seed}.")

    return chkpt, nb_epochs_finished, begin_date

def init(cfg: DictConfig, verbose: bool=True) -> Init:
    if verbose:
        print("Config:")
        print(OmegaConf.to_yaml(cfg))

    gpu    = torch.cuda.is_available()
    device = torch.device('cuda:0' if gpu else 'cpu')
    
    ## Create save & chkpt directories
    save_path, chkpt_path = create_save_directories(cfg)

    ## Load checkpoint if exists
    chkpt, nb_epochs_finished, begin_date = load_chkpt(chkpt_path, device)

    ## DataLoaders
    dl, info = load_dataset_and_make_dataloaders(
        dataset_name=cfg.dataset.name,          # e.g. "FashionMNIST"
        root_dir=cfg.dataset.root_dir,          # directory to store the data 
        batch_size=cfg.dataset.batch_size,      # e.g. 32
        num_workers=cfg.dataset.num_workers,    # can use more workers if GPU is waiting for the batches
        pin_memory=gpu,                         # use pin memory if plan to move the data to GPU
    )
    # XXX: At evaluation time to compute the FID we need to use the
    # "true test set" using train=False.
    
    diffusion = Diffusion(device, info.sigma_data, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max)
    # XXX: info.sigma_data is an estimation of the std based on a "huge" batch

    ## Model and criterion
    model = UNet(
        image_channels=info.image_channels,
        min_channels=cfg.model.min_channels,
        depths=cfg.model.depths,
        cond_channels=cfg.model.cond_channels,
        self_attentions=cfg.model.self_attentions,
        self_attention_bridge=cfg.model.self_attention_bridge,
        nb_heads=cfg.model.nb_heads,
        nb_classes=info.num_classes                             # will +1 fake label for CFG
    )
    criterion = nn.MSELoss()
    model.to(device=device)
    criterion.to(device=device)

    ## Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr)

    ## Load saved model and optimizer state dict if chkpt exists
    if chkpt:
        model.load_state_dict(chkpt["model_state_dict"])
        optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        print("\nSuccessfully loaded model & optimizer state dicts.")

    print(f"\n\nDataset: {cfg.dataset.name}, Using device: {device}")

    return Init(model, optimizer, criterion, diffusion,
                dl, info, device, nb_epochs_finished,
                begin_date, save_path, chkpt_path)

def init_sampling(cfg: DictConfig) -> InitSample:
    seed = torch.random.initial_seed()  # retrieve current seed

    # Initialization
    init_tuple = init(cfg)  # TODO: make it more efficient
    model, diffusion, info, dl = init_tuple.model, init_tuple.diffusion, init_tuple.info, init_tuple.dl
    del init_tuple
    try:
        sampling_method = cfg.common.sampling.method
    except:
        sampling_method = "euler"
    dataset_name = str.lower(cfg.dataset.name)
    cfgscale_str = str(cfg.common.sampling.cfg_scale).replace('.','_')
    path = Path(f"./results/images/{dataset_name}/{sampling_method}/")
    
    # Don't use the checkpoint seed for sampling
    torch.manual_seed(seed)
    return InitSample(model, diffusion, info, dl, sampling_method, path, cfgscale_str)