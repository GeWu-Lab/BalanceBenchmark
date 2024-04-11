from .utils.logger import setup_logger
from .models import create_model
from .trainer import create_trainer
from .utils.train_utils import choose_logger
from .utils.data_utils import create_train_val_dataloader
from types import SimpleNamespace
from .utils.optimizer import create_optimizer
from .utils.scheduler import create_scheduler
from lightning.fabric import Fabric
import lightning as L
from os import path as osp
import os
import torch
import yaml

def train_and_test(args: dict):
    args = SimpleNamespace(**args)
    sys_log = setup_logger(osp.join(args.out_dir, "train.log"))

    log_dir = osp.join(args.out_dir, "logs")
    loggers = [choose_logger(logger_name, log_dir = log_dir, project = args.name, comment = args.log['comment']) for logger_name in args.log['logger_name']]

    args.checkpoint_dir = osp.join(args.out_dir, 'checkpoints')
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for logger in loggers:
        if isinstance(logger, L.fabric.loggers.CSVLogger):
            os.makedirs(os.path.join(log_dir, 'csv'), exist_ok=True)
        elif isinstance(logger, L.fabric.loggers.TensorBoardLogger):
            os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
        elif isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
            os.makedirs(os.path.join(log_dir, 'wandb'), exist_ok=True)

    fabric = Fabric(**(args.fabric),
                    loggers = loggers,
                    )

    if isinstance(fabric.accelerator, L.fabric.accelerators.CUDAAccelerator):
        fabric.print('set float32 matmul precision to high')
        torch.set_float32_matmul_precision('high')

    train_dataloader, val_dataloader = create_train_val_dataloader(fabric, args.dataset)
    
    model = create_model(args.model)
    optimizer = create_optimizer(model, args.train['optimizer'], args.train['parameter'])
    scheduler = create_scheduler(optimizer, args.train['scheduler'])
    trainer = create_trainer(fabric, args.trainer, args.trainer_para)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer.fit(model, train_dataloader, val_dataloader, optimizer, scheduler)

    