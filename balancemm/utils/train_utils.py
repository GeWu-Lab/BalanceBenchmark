import os, glob
from typing import Optional
import subprocess

import torch.nn as nn

from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger

import random
import numpy as np
import torch
import os
def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            total += p.numel()
    return total

def choose_logger(logger_name: str, log_dir, project: Optional[str] = None, comment: Optional[str] = None, *args, **kwargs):
    if logger_name == "csv":
        return CSVLogger(root_dir = log_dir, name = 'csv', *args, **kwargs)
    elif logger_name == "tensorboard":
        logger = TensorBoardLogger(root_dir=log_dir, name='tensorboard',default_hp_metric=False, *args, **kwargs)
        tensorboard_log_dir = os.path.join(log_dir, 'tensorboard')
        # subprocess.Popen(['tensorboard', '--logdir', tensorboard_log_dir])
        
        return logger
    elif logger_name == "wandb":
        return WandbLogger(project = project, save_dir = log_dir, notes = comment, *args, **kwargs)
    else:
        raise ValueError(f"`logger={logger_name}` is not a valid option.")

def get_checkpoint_files(checkpoint_dir):
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    print(f'the checkpoint is {checkpoint_files}')
    return checkpoint_files

def get_newest_path(out_dir):
    folders = [f for f in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, f)) and len(os.listdir(os.path.join(out_dir, f + '/checkpoints')))>0 ]
    # folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    folder = max(folders)
    folder = os.path.join(out_dir, folder + '/checkpoints')
    if folder:
        return folder
    else:
        raise ValueError('there are no pretrained model')
def set_seed(seed):
    """
    Set random seed for training
    
    Args:
        seed (int): random value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set as {seed}")