from memory_profiler import memory_usage
import time
import torch
from thop import profile
from ..models.avclassify_model import BaseClassifierModel
from typing import Callable, Any, Tuple

def get_model_complexity(model, input_size=(1, 3, 224, 224)):
    input = torch.randn(input_size)
    flops, params = profile(model, inputs=(input, ))
    return flops, params

def getallparams(model: BaseClassifierModel) -> int:
    params = 0
    for param in model.parameters():
        params += param.numel()
    return params

def all_in_one_train(trainprocess: callable, model, logger):
    starttime = time.time()
    mem = max(memory_usage(proc=trainprocess))
    endtime = time.time()
    logger.info(f"Training Time: {endtime-starttime:.2f} seconds")
    logger.info(f"Training Peak Mem: {mem:.2f} MiB")
    logger.info(f"Training Params: {getallparams(model)}")
    
    # New metrics
    flops, _ = get_model_complexity(model)
    logger.info(f"FLOPs: {flops/1e9:.2f} G")

def all_in_one_test(testprocess, model, logger):
    teststart = time.time()
    testprocess()
    testend = time.time()
    logger.info(f"Inference Time: {testend-teststart:.2f} seconds")
    logger.info(f"Inference Params: {getallparams(model)}")

def advanced_model_profiler(
    trainprocess: Callable,
    model: torch.nn.Module,
    logger: Any,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    *args,
    **kwargs
):
    """
    Profile the training process and model complexity.

    :param trainprocess: The training function to be profiled
    :param model: The PyTorch model
    :param logger: Logger object for recording information
    :param input_size: Input size for FLOPs calculation (default: (1, 3, 224, 224))
    :param args: Additional positional arguments for trainprocess
    :param kwargs: Additional keyword arguments for trainprocess
    """
    # Measure training time and memory usage
    starttime = time.time()
    mem = max(memory_usage(proc=(trainprocess, args, kwargs)))
    endtime = time.time()

    # Log training time and memory usage
    logger.info(f"Training Time: {endtime-starttime:.2f} seconds")
    logger.info(f"Training Peak Mem: {mem:.2f} MiB")

    # Log model parameters
    total_params = getallparams(model)
    logger.info(f"Training Params: {total_params:,}")

    # Calculate and log FLOPs
    flops, _ = get_model_complexity(model, input_size)
    logger.info(f"FLOPs: {flops/1e9:.2f} G")

    # Calculate and log model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # Size in MB
    logger.info(f"Model Size: {model_size:.2f} MB")

    # Log theoretical memory requirements
    batch_size = input_size[0]
    theoretical_memory = (total_params * 4 + flops * 4) / (1024 * 1024)  # Assuming 4 bytes per parameter and operation
    logger.info(f"Theoretical Memory Requirement (batch size {batch_size}): {theoretical_memory:.2f} MB")

    # You can add more profiling metrics here as needed