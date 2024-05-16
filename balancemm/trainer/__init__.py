import importlib
import os
from os import path as osp
from types import SimpleNamespace
import lightning as L

__all__ = ['create_trainer']

# automatically scan and import trainer modules
# scan all the files under the data folder with '_trainer' in file names
trainer_folder = osp.dirname(osp.abspath(__file__))
trainer_filenames = [
    osp.splitext(v)[0] for v in os.listdir(trainer_folder)
    if v.endswith('_trainer.py')
]

# import all the trainer modules
_trainer_modules = [
    importlib.import_module(f'.{file_name}', package="balancemm.trainer")
    for file_name in trainer_filenames
]

def create_trainer(fabric: L.Fabric ,trainer_opt:dict, para_opt, args, logger):
    # dynamic instantiation
    for module in _trainer_modules:
        trainer_cls = getattr(module, trainer_opt["trainer"], None)
        if trainer_cls is not None:
            break
    if trainer_cls is None:
        raise ValueError(f'trainer {trainer} is not found.')

    trainer = trainer_cls(fabric, para_opt[trainer_opt['name']], para_opt['base'])
    trainer.checkpoint_dir = args.checkpoint_dir

    print(
        f'Trainer {trainer.__class__.__name__} - {trainer_opt["name"]} '
        'is created.')
    logger.info("normal Settings: %s", para_opt['base'])
    logger.info("trainer Settings: %s", para_opt[trainer_opt['name']])
    return trainer