from memory_profiler import memory_usage
import time
import torch
from torchvision.models import resnet18
from thop import profile
from ..models.avclassify_model import BaseClassifierModel

def get_model_complexity(model, input_size=(1, 3, 224, 224)):
    input = torch.randn(input_size)
    flops, params = profile(model, inputs=(input, ))
    return flops, params

def getallparams(model: BaseClassifierModel) -> int:
    params = 0
    for param in model.parameters():
        params += param.numel()
    return params

def all_in_one_train(trainprocess, model, logger):
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