from .utils.logger import setup_logger
from .models import create_model
from .trainer import create_trainer
from .utils.train_utils import choose_logger
from .utils.data_utils import create_train_val_dataloader
from types import SimpleNamespace
from lightning.fabric import Fabric
import lightning as L
from os import path as osp
import os
import torch
import yaml

def train_pipeline(fabric: L.Fabric, config: SimpleNamespace):
    fabric.seed_everything(config.seed, workers = True)

    train_dataloader, val_dataloader = create_train_val_dataloader(fabric, config.dataset)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    num_iter_per_epoch = len(train_dataloader)

    # model will be created directly on device
    with fabric.init_module():
        trainer = create_trainer(fabric, config)

def train_and_test(args: dict):
    args = SimpleNamespace(**args)
    sys_log = setup_logger(args.out_dir + "/train.log")

    log_dir = args.out_dir + "/logs"
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

    fabric = Fabric(accelerator=args.device['accelerator'], 
                    devices=args.device['devices'], 
                    strategy=args.device['strategy'],
                    precision = args.device['precision'],
                    loggers = loggers, 
                    )

    if isinstance(fabric.accelerator, L.fabric.accelerators.CUDAAccelerator):
        fabric.print('set float32 matmul precision to high')
        torch.set_float32_matmul_precision('high')

    try:
        fabric.launch(train_pipeline, args)
    except KeyboardInterrupt:
        for logger in fabric.loggers:
            if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
                logger.experiment.finish(exit_code = 1)
            logger.finalize('aborted')