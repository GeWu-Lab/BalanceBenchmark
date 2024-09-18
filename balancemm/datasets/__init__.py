import importlib
import os
from os import path as osp
from ..utils.parser_utils import find_module
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