from balancemm.utils.logger import setup_logger
from lightning import fabric
import yaml
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
import logging
from .evaluation.modalitys import Calculate_Shapley

def only_test(args: dict):
    args = SimpleNamespace(**args)
    log_dir = osp.join(args.out_dir, "logs")
    print("logg:{}".format(log_dir))
    # loggers = [choose_logger(logger_name, log_dir = log_dir, project = args.name, comment = args.log['comment']) for logger_name in args.log['logger_name']]
    logger = logging.getLogger(__name__)
    args.checkpoint_dir = osp.join(args.out_dir, 'checkpoints')
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    file_handler = logging.FileHandler(args.out_dir + '/test.log') 
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    loggers = [logger]
    logger.setLevel(logging.DEBUG)
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

    train_dataloader, val_dataloader, test_dataloader = create_train_val_dataloader(fabric, args)
    args.trainer['checkpoint_dir'] = args.checkpoint_dir ##
    model = create_model(args.model)
    if hasattr(args.trainer, 'name') and args.trainer['name'] == 'GBlending':
        args.model['type'] = args.model['type'] + '_gb'
        model = create_model(args.model)
        temp_model = create_model(args.model)
    # optimizer = create_optimizer(model, args.train['optimizer'], args.train['parameter'])
    # scheduler = create_scheduler(optimizer, args.train['scheduler'])
    trainer = create_trainer(fabric, args.Main_config, args.trainer, args, logger)
    device = args.Main_config['device']
    if device == '':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + args.Main_config['device'])
    model.to(device)
    model.device = device
    check_point = torch.load(args.check_point_path)
    model.load_state_dict(check_point['model'])
    trainer.max_epochs = 1
    trainer.fit(model, None, test_dataloader, None, None, logger)
    Calculate_Shapley(trainer = trainer, model = model, CalcuLoader = test_dataloader, logger= logger)

def all_test(args: dict):
    dict_args = args
    args = SimpleNamespace(**args)
    model = args.Main_config['model']
    trainer = args.Main_config['trainer']
    dataset = args.Main_config['dataset']
    checkpoint_rootdir = f"./experiments/{model}_{trainer}_{dataset}/train_and_test"
    for root, dirs, files in os.walk(checkpoint_rootdir):
        if os.path.basename(root).startswith('train_20241104') or os.path.basename(root).startswith('train_20241105') or os.path.basename(root).startswith('train_20241106'):
            checkpoint_path = os.path.join(root, 'checkpoints', 'epoch_normal.ckpt')
            if os.path.isfile(checkpoint_path):
                print(f"找到checkpoint: {checkpoint_path}")
            else:
                continue
            args.out_dir = root.replace("train_and_test", "all_test")
            print(args.out_dir)
        
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
            _, _, test_dataloader = create_train_val_dataloader(fabric, args)
            args.trainer['checkpoint_dir'] = args.checkpoint_dir ##
            model = create_model(args.model)
            trainer = create_trainer(fabric, args.Main_config, args.trainer, args, logger,tb_logger)
            model.to(device)
            model.device = device
                
            # 记录开始时间
            logger.info('Use the best model to Test')
            model.eval()
            best_state = torch.load(checkpoint_path)
            model.load_state_dict(best_state['model'])
            _, Metrics_res = trainer.val_loop(model, test_dataloader)
            test_acc = Metrics_res['acc']['output']
            info = ''
            output_info = ''
            for metircs in sorted(Metrics_res.keys()):
                if metircs == 'acc':
                    valid_acc = Metrics_res[metircs]
                    for modality in sorted(valid_acc.keys()):
                        tag = "valid_acc"
                        if modality == 'output':
                            output_info += f"test_acc: {valid_acc[modality]}"

                        else:
                            info += f", acc_{modality}: {valid_acc[modality]}"
                        
                            
                if metircs == 'f1':
                    valid_f1 = Metrics_res[metircs]
                    for modality in sorted(valid_f1.keys()):
                        tag = "valid_f1"
                        if modality == 'output':
                            output_info += f", test_f1: {valid_f1[modality]}"

                        else:
                            info += f", f1_{modality}: {valid_f1[modality]}"
                        
            info = output_info+ ', ' + info
                
            logger.info(info)
            for handler in logger.handlers:
                handler.flush()
            logger.info(f'The best test acc is : {test_acc}')
            print(f'The best test acc is : {test_acc}')
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        