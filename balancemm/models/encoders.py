import csv
import random
from functools import partialmethod

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os

    
    
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from torch import nn

class text_encoder(nn.Module):
    def __init__(self, dim_text_repr=768):
        super().__init__()
        config = BertConfig()
        self.textEncoder= BertModel(config).from_pretrained('')    

    def forward(self, x):
        text = x
        hidden_states = self.textEncoder(**text)  # B, T, dim_text_repr
        e_i = F.dropout(hidden_states[1]) 
        return e_i

class image_encoder(nn.Module):
    def __init__(self,model_arch,num_classes=10,weights="IMAGENET1K_V1",device="cuda"):
        super().__init__()
        self.model_arch=model_arch
        self.device=device
        self.num_classes=num_classes
        print(model_arch)
        self.model=torch.hub.load("pytorch/vision", self.model_arch, weights=weights).to(self.device)
        # print(self.model.parameters)
        self.model.heads = nn.Sequential()

    def forward(self,x):
        y=self.model(x)
        return y
