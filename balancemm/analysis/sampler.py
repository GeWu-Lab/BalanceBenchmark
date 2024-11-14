import torch
import torch.nn as nn
def batch_sampler(dataloader, class_nums):
    for idx, batch in dataloader:
        for i in len(batch['label']):
            