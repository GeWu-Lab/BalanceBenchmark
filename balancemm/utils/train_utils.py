import os, glob
from typing import Optional

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
        return TensorBoardLogger(root_dir = log_dir, name = 'tensorboard', *args, **kwargs)
    elif logger_name == "wandb":
        return WandbLogger(project = project, save_dir = log_dir, notes = comment, *args, **kwargs)
    else:
        raise ValueError(f"`logger={logger_name}` is not a valid option.")

def get_checkpoint_files(checkpoint_dir):
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    return checkpoint_files

def set_seed(seed):
    """
    设置整个训练过程的随机种子
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置 CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 Python 的 hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set as {seed}")