import importlib
import os
from os import path as osp
from ..utils.parser_utils import find_module
import torch.nn as nn
import torch
__all__ = ['find_encoder', 'create_encoder']
from .pretrained_encoder import text_encoder
from torchvision.models import vit_b_16, vit_h_14


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

def create_encoders(encoder_opt: dict[str, dict])->dict[str, nn.Module]:
    modalitys = encoder_opt.keys()
    encoders = {}
    for modality in modalitys:
        pre_train = encoder_opt[modality]['if_pretrain']
        path = encoder_opt[modality]['pretrain_path']
        name = encoder_opt[modality]['name']
        if name == "ViT_B":
            if pre_train:
                encoders[modality] = vit_b_16(path)
            else:
                encoders[modality] = vit_b_16()
            continue
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