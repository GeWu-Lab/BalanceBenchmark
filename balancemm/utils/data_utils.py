import torch.utils.data
from torchvision import datasets, transforms
from types import SimpleNamespace
from ..datasets import create_dataset
import lightning as L

def create_train_val_dataloader(fabric: L.Fabric, config: dict):
    config = SimpleNamespace(**config)
    
    train_dataset = create_dataset(config.train)
    val_dataset = create_dataset(config.val)
    
    config_dataloader = SimpleNamespace(**config.dataloader)
    if config_dataloader.fast_run == True:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(config_dataloader.eff_batch_size*4)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(config_dataloader.eff_batch_size*2)))
    
    # print len of datasets
    fabric.print(f"Train dataset: {train_dataset.__class__.__name__} - {config.train['name']}, {len(train_dataset)} samples")
    fabric.print(f"Val dataset: {val_dataset.__class__.__name__} - {config.val['name']}, {len(val_dataset)} samples")

    if (not hasattr(config_dataloader, 'batch_size')):
        config_dataloader.batch_size = round(config_dataloader.eff_batch_size/fabric.world_size) # using the effective batch_size to calculate the batch_size per gpu
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,  batch_size=config_dataloader.batch_size, shuffle=config_dataloader.shuffle, drop_last = config_dataloader.drop_last, num_workers = config_dataloader.num_workers, multiprocessing_context='spawn', pin_memory = config_dataloader.pin_memory)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,  batch_size=config_dataloader.batch_size, drop_last = config_dataloader.drop_last, num_workers = config_dataloader.num_workers, multiprocessing_context='spawn', pin_memory = config_dataloader.pin_memory)

    # print batchsize and len
    fabric.print(f"Train dataloader: {len(train_dataloader)} batches, {len(train_dataloader.dataset)} samples")
    fabric.print(f"Val dataloader: {len(val_dataloader)} batches, {len(val_dataloader.dataset)} samples")

    return train_dataloader, val_dataloader