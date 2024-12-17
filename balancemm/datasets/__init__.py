import importlib
import os
from os import path as osp
from ..utils.parser_utils import find_module
from torch.utils.data import Dataset
import torch
import numpy as np
from math import factorial
from .KS_dataset import KSDataset
from .cremad_dataset import CremadDataset
from .food_dataset import UMPC_FoodDataset
from .Mosei_dataset import MoseiDataset
from .balance_dataset import BalanceDataset
from .ucf101_dataset import UCFDataset
import pandas as pd
__all__ = ['find_dataset', 'create_dataset']

# ----------------------
# Dynamic instantiation
# ----------------------

# import dataset modules from those with '_dataset' in file names
_data_folder = osp.dirname(osp.abspath(__file__))
_dataset_filenames = [
    osp.splitext(v)[0] for v in os.listdir(_data_folder)
    if v.endswith('_dataset.py')
]
_dataset_modules = [
    importlib.import_module(f'.{file_name}', package="balancemm.datasets")
    for file_name in _dataset_filenames
]

# find dataset from dataset_opt
def find_dataset(dataset_name: str) -> object:
    dataset_cls = find_module(_dataset_modules, dataset_name, 'Dataset')
    return dataset_cls

# create dataset from dataset_opt
def create_dataset(dataset_opt: dict, mode: str):
    if 'name' not in dataset_opt:
        raise ValueError('Dataset name is required.')
    dataset_cls = find_dataset(dataset_opt['name'])
    dataset_opt['mode'] = mode
    dataset = dataset_cls(dataset_opt)

    print (f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} is created.')
    return dataset

class SampleDataset(Dataset):

    def __init__(self, original_dataset, zero_modality_samples):
        self.dataset = original_dataset
        self.zero_modality_samples = zero_modality_samples

        for attr_name in dir(original_dataset):
            if not attr_name.startswith('__'):
                try:
                    setattr(self, attr_name, getattr(original_dataset, attr_name))
                except AttributeError:
                    continue
    
    def _handle_mosei_data(self, idx, data):

        if idx in self.zero_modality_samples:
            for modality in self.zero_modality_samples[idx]:
                if modality == 'text':
                    data['text'] = torch.zeros_like(data['text'])
                elif modality == 'audio':
                    data['audio'] = torch.zeros_like(data['audio'])
                elif modality == 'visual':
                    data['visual'] = torch.zeros_like(data['visual'])
        return data

    def _handle_av_data(self, idx, data):

        if idx in self.zero_modality_samples:
            for modality in self.zero_modality_samples[idx]:
                if modality == 'audio':
                    if isinstance(data['audio'], torch.Tensor):
                        data['audio'] = torch.zeros_like(data['audio'])
                    else:
                        data['audio'] = np.zeros_like(data['audio'])
                elif modality == 'visual':
                    data['visual'] = torch.zeros_like(data['visual'])
        return data
    
    def _handle_ucf_data(self, idx, data):
    
        if idx in self.zero_modality_samples:
            for modality in self.zero_modality_samples[idx]:
                if modality == 'visual':
                    data['visual'] = torch.zeros_like(data['visual'])
                elif modality == 'flow':
                    data['flow'] = torch.zeros_like(data['flow'])
        return data

    def _handle_food_data(self, idx, data):

        if idx in self.zero_modality_samples:
            for modality in self.zero_modality_samples[idx]:
                if modality == 'text':
                    data['text'] = torch.zeros_like(data['text'])
                elif modality == 'visual':
                    data['visual'] = torch.zeros_like(data['visual'])
        return data


    def __getitem__(self, idx):

        data = self.dataset.__getitem__(idx)
        
        if isinstance(self.dataset, MoseiDataset):
            return self._handle_mosei_data(idx, data)
        elif isinstance(self.dataset, (KSDataset, BalanceDataset, CremadDataset)):
            return self._handle_av_data(idx, data)
        elif isinstance(self.dataset, UMPC_FoodDataset):
            return self._handle_food_data(idx, data)
        elif isinstance(self.dataset, UCFDataset):
            return self._handle_food_data(idx, data)
        else:
            raise ValueError('Dataset name is required.')
            

    def __len__(self):
        return len(self.dataset)
    
def create_dataset_sample_level(dataset_opt: dict, mode: str, contribution: dict, modality_list: list):
    if 'name' not in dataset_opt:
        raise ValueError('Dataset name is required.')
    
    dataset_cls = find_dataset(dataset_opt['name'])
    dataset_opt['mode'] = mode
    dataset = dataset_cls(dataset_opt)

    zero_modality_samples = {}
    if isinstance(dataset,MoseiDataset):
        new_vision = [dataset.vision]
        new_text = [dataset.text]
        new_audio = [dataset.audio]
        new_labels = [dataset.labels]
        if dataset.meta is not None:
            new_meta = [dataset.meta]  
        else:
            new_meta = None
        
    elif isinstance(dataset,UCFDataset):
        new_data = dataset.data.copy()
        new_data2class = dataset.data2class.copy()
    elif isinstance(dataset,UMPC_FoodDataset):
        new_data = dataset.data.copy()
    else:
        new_data = dataset.data.copy()
        new_label = dataset.label.copy() if hasattr(dataset, 'label') else None

    for i in range(len(dataset)-len(dataset) % 64): # batch_size
        for modality in modality_list:
            shapley_value = contribution[modality][i]
            
            if 0.4 < shapley_value < 1:
                copies = 1
            elif -0.1 < shapley_value < 0.4:
                copies = 2
            elif shapley_value < -0.1:
                copies = 3
            else:
                continue
            for _ in range(copies):
                
                if isinstance(dataset,MoseiDataset):
                    new_idx = len(new_vision)
                    new_vision.append(dataset.vision[i:i+1])
                    new_text.append(dataset.text[i:i+1])
                    new_audio.append(dataset.audio[i:i+1])
                    new_labels.append(dataset.labels[i:i+1])
                    if new_meta is not None: 
                        new_meta.append(dataset.meta[i:i+1])
                elif isinstance(dataset,(KSDataset, BalanceDataset, CremadDataset)):
                    new_idx = len(new_data)
                    new_data.append(dataset.data[i])
                    if new_label is not None:
                        new_label.append(dataset.label[i])
                elif isinstance(dataset,UMPC_FoodDataset):
                    new_idx = len(new_data)
                    new_data = pd.concat([new_data, dataset.data.iloc[i:i+1]], ignore_index=True)
                elif isinstance(dataset,UCFDataset):
                    new_idx = len(new_data)
                    datum = dataset.data[i]
                    new_data.append(datum)
                    new_data2class[datum] = dataset.data2class[datum]
                if new_idx not in zero_modality_samples:
                    zero_modality_samples[new_idx] = []
                for modality_another in modality_list:
                    if modality_another == modality:
                        continue
                    zero_modality_samples[new_idx].append(modality_another)
             
    if isinstance(dataset,MoseiDataset):
        dataset.vision = torch.cat(new_vision, dim=0)
        dataset.text = torch.cat(new_text, dim=0)
        dataset.audio = torch.cat(new_audio, dim=0)
        dataset.labels = torch.cat(new_labels, dim=0)
        if new_meta is not None:
            dataset.meta = np.concatenate(new_meta, axis=0)
    elif isinstance(dataset,(KSDataset, BalanceDataset, CremadDataset)):
        dataset.data = new_data
        if new_label is not None:
            dataset.label = new_label
    elif isinstance(dataset, UMPC_FoodDataset):
        dataset.data = new_data
    elif isinstance(dataset,UCFDataset):
        dataset.data = new_data
        dataset.data2class = new_data2class  
    
    sample_dataset = SampleDataset(dataset, zero_modality_samples)
    return sample_dataset


def create_dataset_modality_level(dataset_opt: dict, mode: str, contribution: dict, part_ratio: float, modality_list: list,func = 'linear'):
    if 'name' not in dataset_opt:
        raise ValueError('Dataset name is required.')
    
    dataset_cls = find_dataset(dataset_opt['name'])
    dataset_opt['mode'] = mode
    dataset = dataset_cls(dataset_opt)

    zero_modality_samples = {}

    length = len(dataset)-len(dataset) % 64
    num = int(length * part_ratio)
    choice = np.random.choice(length, num)
    print(len(choice))
    part_contribution = {modality: 0.0 for modality in modality_list} 
    
    for i in choice:
        for modality in modality_list:
            part_contribution[modality] += contribution[modality][i] 
    for modality in modality_list:
        part_contribution[modality] /= num
    
    gap = {modality:1.0-part_contribution[modality] for modality in modality_list}
    if func == 'linear':
        difference = 0.0
        for modality in modality_list:
            for modality_another in  modality_list:
                if modality_another == modality:
                    continue
                
                difference += abs(gap[modality] - gap[modality_another])
        difference /= factorial(len(modality_list))
        difference = (difference / 3 * 2 ) 
    elif func == 'tanh':
        tanh = torch.nn.Tanh()
        difference = 0.0
        for modality in modality_list:
            for modality_another in  modality_list:
                if modality_another == modality:
                    continue
                
                difference += abs(gap(modality) - gap(modality_another))
        difference /= factorial(len(modality_list))
        difference = tanh(torch.tensor((difference / 3 * 2 )))
    elif func == 'square':
        difference = 0.0
        for modality in modality_list:
            for modality_another in  modality_list:
                if modality_another == modality:
                    continue
                
                difference += abs(gap(modality) - gap(modality_another))
        difference /= factorial(len(modality_list))
        difference = (difference / 3 * 2 ) ** 1.5
    resample_num = int(difference * length)
    sample_choice = np.random.choice(length, resample_num)
    
    if isinstance(dataset,MoseiDataset):
        new_vision = [dataset.vision]
        new_text = [dataset.text]
        new_audio = [dataset.audio]
        new_labels = [dataset.labels]
        if dataset.meta is not None:
            new_meta = [dataset.meta]  
        else:
            new_meta = None
        
    elif isinstance(dataset,UCFDataset):
        new_data = dataset.data.copy()
        new_data2class = dataset.data2class.copy()
    elif isinstance(dataset,UMPC_FoodDataset):
        new_data = dataset.data.copy()
    else:
        new_data = dataset.data.copy()
        new_label = dataset.label.copy() if hasattr(dataset, 'label') else None

    for i in sample_choice:
        for modality in modality_list:
            if isinstance(dataset,MoseiDataset):
                new_idx = len(new_vision)
                new_vision.append(dataset.vision[i:i+1])
                new_text.append(dataset.text[i:i+1])
                new_audio.append(dataset.audio[i:i+1])
                new_labels.append(dataset.labels[i:i+1])
                if new_meta is not None: 
                    new_meta.append(dataset.meta[i:i+1])

            elif isinstance(dataset,(KSDataset, BalanceDataset, CremadDataset)):
                new_idx = len(new_data)
                new_data.append(dataset.data[i])
                if new_label is not None:
                    new_label.append(dataset.label[i])
            elif isinstance(dataset,UMPC_FoodDataset):
                new_idx = len(new_data)
                new_data = pd.concat([new_data, dataset.data.iloc[i:i+1]], ignore_index=True)
            elif isinstance(dataset,UCFDataset):
                new_idx = len(new_data)
                datum = dataset.data[i]
                new_data.append(datum)
                new_data2class[datum] = dataset.data2class[datum]
            if new_idx not in zero_modality_samples:
                zero_modality_samples[new_idx] = []
            for modality_another in modality_list:
                if modality_another == modality:
                    continue
                if gap[modality] > gap[modality_another]:
                    zero_modality_samples[new_idx].append(modality_another)
    if isinstance(dataset,MoseiDataset):
        dataset.vision = torch.cat(new_vision, dim=0)
        dataset.text = torch.cat(new_text, dim=0)
        dataset.audio = torch.cat(new_audio, dim=0)
        dataset.labels = torch.cat(new_labels, dim=0)
        if new_meta is not None:
            dataset.meta = np.concatenate(new_meta, axis=0)
    elif isinstance(dataset, UMPC_FoodDataset):
        dataset.data = new_data
    elif isinstance(dataset,UCFDataset):
        dataset.data = new_data
        dataset.data2class = new_data2class 
    elif isinstance(dataset,(KSDataset, BalanceDataset, CremadDataset)):
        dataset.data = new_data
        if new_label is not None:
            dataset.label = new_label
    
    sample_dataset = SampleDataset(dataset, zero_modality_samples)
    return sample_dataset