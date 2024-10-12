from memory_profiler import memory_usage
import time
import torch
from thop import profile
from ..models.avclassify_model import BaseClassifierModel
from typing import Callable, Any, Tuple
## flops库 
## 命名规范
from thop import profile
import functools

class FLOPsMonitor:
    def __init__(self):
        self.total_flops = 0
        self.forward_flops = 0
        self.backward_flops = 0

    def update(self, flops, operation='total'):
        if operation == 'forward':
            self.forward_flops += flops
        elif operation == 'backward':
            self.backward_flops += flops
        self.total_flops += flops

    def report(self, logger):
        print(f"Total FLOPs: {self.total_flops}")
        print(f"Forward FLOPs: {self.forward_flops}")
        print(f"Backward FLOPs: {self.backward_flops}")
        logger.info(f"Total FLOPs: {self.total_flops}")
        logger.info(f"Forward FLOPs: {self.forward_flops}")
        logger.info(f"Backward FLOPs: {self.backward_flops}")
        
        
def get_flops(model, input_sample):
    with torch.no_grad():
        flops, params = profile(model, inputs= input_sample)
    return flops, params