from .utils.logger import setup_logger
from .models import create_model
from .trainer import create_trainer
from .utils.train_utils import choose_logger
import subprocess
from .utils.data_utils import create_train_val_dataloader
from types import SimpleNamespace
from .utils.optimizer import create_optimizer
from .utils.scheduler import create_scheduler
from lightning.fabric import Fabric
import lightning as L
from os import path as osp
import os
import torch
import logging
from datetime import datetime
from .evaluation.modalitys import Calculate_Shapley
def train_and_test(args: dict):
    dict_args = args
    args = SimpleNamespace(**args)

    if args.trainer['name'] == 'unimodal':
        args.out_dir = args.out_dir.replace('unimodalTrainer','unimodalTrainer_' + list(args.model['encoders'].keys())[0])

    log_dir = osp.join(args.out_dir, "logs")
    print("logg:{}".format(log_dir))
    loggers_online = [choose_logger(logger_name, log_dir = log_dir, project = args.name, comment = args.log['comment']) for logger_name in args.log['logger_name']]
    logger = logging.getLogger(__name__)
    args.checkpoint_dir = osp.join(args.out_dir, 'checkpoints')
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    file_handler = logging.FileHandler(args.out_dir + '/training.log') 
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    loggers = [logger]
    logger.setLevel(logging.DEBUG)
    logger.info(dict_args)
    for tb_logger in loggers_online:
        if isinstance(tb_logger, L.fabric.loggers.CSVLogger):
            os.makedirs(os.path.join(log_dir, 'csv'), exist_ok=True)
        elif isinstance(tb_logger, L.fabric.loggers.TensorBoardLogger):
            os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
        elif isinstance(tb_logger, L.pytorch.loggers.wandb.WandbLogger):
            os.makedirs(os.path.join(log_dir, 'wandb'), exist_ok=True)
    if tb_logger:
        tb_logger.log_hyperparams(dict_args)
    fabric = Fabric(**(args.fabric),
                    loggers = loggers,
                    )
    if isinstance(fabric.accelerator, L.fabric.accelerators.CUDAAccelerator):
        fabric.print('set float32 matmul precision to high')
        torch.set_float32_matmul_precision('high')
    device = args.Main_config['device']
    if device == '':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + args.Main_config['device'])
    args.model['device'] = device
    train_dataloader, val_dataloader, test_dataloader = create_train_val_dataloader(fabric, args)
    args.trainer['checkpoint_dir'] = args.checkpoint_dir ##
    model = create_model(args.model)
    optimizer = create_optimizer(model, args.train['optimizer'], args.train['parameter'])
    scheduler = create_scheduler(optimizer, args.train['scheduler'])
    trainer = create_trainer(fabric, args.Main_config, args.trainer, args, logger,tb_logger)
    model.to(device)
    model.device = device
        
    # 记录开始时间
    start_time = datetime.now()
    if args.trainer['name'] == 'GBlendingTrainer':
        temp_model = create_model(args.model)
        temp_model.to(device)
        temp_model.device = device
        trainer.fit(model, temp_model,train_dataloader, val_dataloader, optimizer, scheduler, logger,tb_logger)
    else :
        trainer.fit(model, train_dataloader, val_dataloader, optimizer, scheduler, logger,tb_logger) 
    end_time = datetime.now()
    total_time = end_time - start_time
    total_time = total_time.total_seconds() / 3600
    logger.info("Training time :{:.2f}".format(total_time))
    logger.info('Use the best model to Test')
    model.eval()
    best_state = torch.load(args.checkpoint_dir+ '/epoch_normal.ckpt')
    model.load_state_dict(best_state['model'])
    trainer.val_loop(model, test_dataloader)
    logger.info('Calculate the shapley value of best model')
    Calculate_Shapley(trainer = trainer, model = model, CalcuLoader = test_dataloader, logger= logger)

    # 多卡（模型名字区别）

    