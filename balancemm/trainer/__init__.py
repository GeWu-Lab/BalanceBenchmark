import importlib
import os
from os import path as osp
from types import SimpleNamespace
import lightning as L
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger

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

def create_trainer(fabric: L.Fabric ,trainer_opt:dict, para_opt, args, logger,tb_logger):
    # dynamic instantiation
    for module in _trainer_modules:
        trainer_cls = getattr(module, trainer_opt["trainer"], None)
        if trainer_cls is not None:
            break
    if trainer_cls is None:
        raise ValueError(f'trainer {trainer} is not found.')
    para_opt['base_para']['logger'] = logger
    if args.trainer['name'] != 'UMT' and args.trainer['name'] != 'LinearProbe' and args.trainer['name'] != "MLA" and args.trainer['name'] != "OPM" and args.trainer["name"] != "Sample":
        trainer = trainer_cls(fabric, para_opt, para_opt['base_para'])
    else:
        trainer = trainer_cls(fabric, para_opt, para_opt['base_para'], args)
    trainer.checkpoint_dir = args.checkpoint_dir

    print(
        f'Trainer {trainer.__class__.__name__} - {trainer_opt["trainer"]} '
        'is created.')
    para_opt['name'] = trainer_opt["trainer"]
    # logger.info("normal Settings: %s", para_opt)
    logger.info("trainer Settings: %s", para_opt)
    if isinstance(tb_logger, TensorBoardLogger):
        tb_logger.log_hyperparams(trainer_opt)  
        tb_logger.experiment.add_text("Trainer Setup", f"Trainer: {trainer_opt['trainer']}")
    return trainer