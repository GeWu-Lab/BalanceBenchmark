import importlib
import os
from os import path as osp
from types import SimpleNamespace
from ..utils.parser_utils import find_module
__all__ = ['find_model', 'create_model']


# import model modules from those with '_model' in file names
_model_folder = osp.dirname(osp.abspath(__file__))
_model_filenames = [
    osp.splitext(v)[0] for v in os.listdir(_model_folder)
    if v.endswith('_model.py')
]
_model_modules = [
    importlib.import_module(f'.{file_name}', package="balancemm.models")
    for file_name in _model_filenames
]

# find model from model_opt
def find_model(model_name: str) -> object:
    model_cls = find_module(_model_modules, model_name, 'Model')
    return model_cls

# create model from model_opt
def create_model(model_opt: dict):
    if 'type' not in model_opt:
        raise ValueError('Model type is required.')
    model_cls = find_model(model_opt['type'])
    model = model_cls(model_opt)

    print(
        f'Model {model.__class__.__name__} - {model_opt["type"]} is created.')
    return model