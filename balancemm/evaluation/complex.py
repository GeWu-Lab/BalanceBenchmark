from memory_profiler import memory_usage
import time
import torch
from thop import profile
from ..models.avclassify_model import BaseClassifierModel
from typing import Callable, Any, Tuple
## flops库 
## 命名规范
from torch.profiler import profile, record_function, ProfilerActivity
import functools

def profile_flops(logger=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 尝试获取 logger
            nonlocal logger
            if logger is None:
                logger = getattr(self, 'logger', None)
            
            if logger is None:
                # 如果还是没有 logger，可以创建一个默认的或抛出异常
                raise ValueError("No logger found")

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         with_flops=True,
                         profile_memory=True) as prof:
                with record_function(func.__name__):
                    result = func(self, *args, **kwargs)
            
            for event in prof.key_averages():
                if event.key == "cuda":  # 或使用 "cpu" 如果你想看 CPU 内存
                    max_memory = max(max_memory, event.cpu_memory_usage, event.cuda_memory_usage)
                else :
                    max_memory = max(max_memory, event.cpu_memory_usage)

            # print(f"Max memory usage in {func.__name__}: {max_memory / (1024 * 1024):.2f} MB")

            # 计算并记录 FLOPs
            total_flops = sum(event.flops for event in prof.events())
            logger.info('Total flops is : {:0}, memory usage is {:1} GB'.format(total_flops, max_memory/(1024**3)))
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

            # ... 其他日志记录逻辑 ...

            return result
        return wrapper
    return decorator