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
    def __init__(self, output_dim=1024):
        super().__init__()
        config = BertConfig()
        self.textEncoder= BertModel(config).from_pretrained('')    
        self.linear = nn.Linear(config.hidden_size, output_dim)
        
    def forward(self, x):
        text = x.squeeze(1)
        hidden_states = self.textEncoder(text)  
        e_i = self.linear(hidden_states[1]) 
        e_i = F.dropout(e_i)
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
