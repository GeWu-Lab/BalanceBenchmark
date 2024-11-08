import torch.utils.data
from torchvision import datasets, transforms
from types import SimpleNamespace
from ..datasets import create_dataset
import lightning as L
import numpy as np
import random
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_train_val_dataloader(fabric: L.Fabric, config: dict):
    # config = SimpleNamespace(**config)
    train_dataset = create_dataset(config.dataset, 'train')
    if config.dataset.get('validation', False):
        val_dataset = create_dataset(config.dataset, 'valid')
        test_dataset = create_dataset(config.dataset, 'test')
    else:
        val_dataset = create_dataset(config.dataset, 'test')
        test_dataset = val_dataset
    
    config_dataloader = SimpleNamespace(**config.dataloader)
    if config_dataloader.fast_run == True:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(config_dataloader.eff_batch_size*4)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(config_dataloader.eff_batch_size*2)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(config_dataloader.eff_batch_size*2)))
    # print len of datasets
    fabric.print(f"Train dataset: {train_dataset.__class__.__name__} - {config.Train['dataset']}, {len(train_dataset)} samples")
    fabric.print(f"Val dataset: {val_dataset.__class__.__name__} - {config.Val['dataset']}, {len(val_dataset)} samples")

    if (not hasattr(config_dataloader, 'batch_size') or config_dataloader.batch_size == -1):
        config_dataloader.batch_size = round(config_dataloader.eff_batch_size/fabric.world_size) # using the effective batch_size to calculate the batch_size per gpu

    train_dataloader = torch.utils.data.DataLoader(train_dataset,  
                                                   batch_size=config_dataloader.batch_size, 
                                                   shuffle=config_dataloader.shuffle, 
                                                   drop_last = config_dataloader.drop_last,
                                                     num_workers = config_dataloader.num_workers, 
                                                     multiprocessing_context='spawn', 
                                                     pin_memory = config_dataloader.pin_memory)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,  batch_size=config_dataloader.batch_size, drop_last = config_dataloader.drop_last, 
                                                 num_workers = config_dataloader.num_workers, 
                                                multiprocessing_context='spawn', 
                                                pin_memory = config_dataloader.pin_memory)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,  batch_size=config_dataloader.batch_size, drop_last = config_dataloader.drop_last, 
                                                 num_workers = config_dataloader.num_workers, 
                                                multiprocessing_context='spawn', 
                                                pin_memory = config_dataloader.pin_memory)

    if config.Train['dataset'] == 'UMPC_Food':
        g = torch.Generator()
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config_dataloader.batch_size, 
                                                shuffle=config_dataloader.shuffle, 
                                                drop_last = config_dataloader.drop_last,
                                                num_workers = config_dataloader.num_workers, 
                                                multiprocessing_context='spawn', 
                                                pin_memory = config_dataloader.pin_memory,
                                               worker_init_fn=worker_init_fn,
                                               generator=g)

        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config_dataloader.batch_size, 
                                                shuffle=config_dataloader.shuffle, 
                                                drop_last = config_dataloader.drop_last,
                                                num_workers = config_dataloader.num_workers, 
                                                multiprocessing_context='spawn', 
                                                pin_memory = config_dataloader.pin_memory,
                                                worker_init_fn=worker_init_fn,
                                                generator=g)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=config_dataloader.batch_size, 
                                                shuffle=config_dataloader.shuffle, 
                                                drop_last = config_dataloader.drop_last,
                                                num_workers = config_dataloader.num_workers, 
                                                multiprocessing_context='spawn', 
                                                pin_memory = config_dataloader.pin_memory,
                                                worker_init_fn=worker_init_fn,
                                                generator=g)
    # print batchsize and len
    fabric.print(f"Train dataloader: {len(train_dataloader)} batches, {len(train_dataloader.dataset)} samples")
    fabric.print(f"Val dataloader: {len(val_dataloader)} batches, {len(val_dataloader.dataset)} samples")

    return train_dataloader, val_dataloader, test_dataloader