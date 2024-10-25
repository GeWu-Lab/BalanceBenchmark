import importlib
import os
from os import path as osp
from ..utils.parser_utils import find_module
import torch.nn as nn
import torch
__all__ = ['find_encoder', 'create_encoder']
from .pretrained_encoder import text_encoder

# ----------------------
# Dynamic instantiation
# ----------------------

# import encoder modules from those with '_encoder' in file names
_encoder_folder = osp.dirname(osp.abspath(__file__))
_encoder_filenames = [
    osp.splitext(v)[0] for v in os.listdir(_encoder_folder)
    if v.endswith('_encoder.py')
]
_encoder_modules = [
    importlib.import_module(f'.{file_name}', package="balancemm.encoders")
    for file_name in _encoder_filenames
]

# find encoder from encoder_opt
def find_encoder(encoder_name: str) -> object:
    encoder_cls = find_module(_encoder_modules, encoder_name, 'Encoder')
    return encoder_cls

# create encoder from encoder_opt
# def create_encoder(encoder_opt: dict):
#     if 'name' not in encoder_opt:
#         raise ValueError('encoder name is required.')
#     encoder_cls = find_encoder(encoder_opt['name'])
#     encoder = encoder_cls(encoder_opt)

#     print (f'Encoder {encoder.__class__.__name__} - {encoder_opt["name"]} is created.')
#     return encoder
def create_encoders(encoder_opt: dict[str, dict])->dict[str, nn.Module]:
    modalitys = encoder_opt.keys()
    encoders = {}
    for modality in modalitys:
        pre_train = encoder_opt[modality]['if_pretrain']
        path = encoder_opt[modality]['pretrain_path']
        name = encoder_opt[modality]['name']
        del encoder_opt[modality]['pretrain_path']
        del encoder_opt[modality]['if_pretrain']
        encoder = find_encoder(encoder_opt[modality]['name'])
        del encoder_opt[modality]['name']
        encoders[modality] = encoder(**encoder_opt[modality])
        encoder_opt[modality]['name'] = name
        if pre_train:
            if modality == 'text':
                encoders[modality] = text_encoder()
            else:
                state = torch.load(path)
                if modality == 'flow':
                    del state['conv1.weight']
                encoders[modality].load_state_dict(state, strict=False)
            print('pretrain load finish')
        encoder_opt[modality]['if_pretrain'] = pre_train
        encoder_opt[modality]['pretrain_path'] = path
        print (f'Encoder {name} - {modality} is created.')
    return encoders