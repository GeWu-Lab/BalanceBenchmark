import importlib
import os
from os import path as osp
from ..utils.parser_utils import find_module
__all__ = ['find_encoder', 'create_encoder']

# ----------------------
# Dynamic instantiation
# ----------------------

# import dataset modules from those with '_dataset' in file names
_encoder_folder = osp.dirname(osp.abspath(__file__))
_encoder_filenames = [
    osp.splitext(v)[0] for v in os.listdir(_encoder_folder)
    if v.endswith('_encoder.py')
]
_encoder_modules = [
    importlib.import_module(f'.{file_name}', package="balancemm.encoders")
    for file_name in _encoder_filenames
]

# find dataset from dataset_opt
def find_encoder(dataset_name: str) -> object:
    dataset_cls = find_module(_encoder_modules, dataset_name, 'Dataset')
    return dataset_cls

# create dataset from dataset_opt
def create_encoder(dataset_opt: dict):
    if 'name' not in dataset_opt:
        raise ValueError('Dataset name is required.')
    dataset_cls = find_encoder(dataset_opt['name'])
    dataset = dataset_cls(dataset_opt)

    print (f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} is created.')
    return dataset