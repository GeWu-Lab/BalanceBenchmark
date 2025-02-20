from typing import Mapping
from lightning import LightningModule
from torch.optim.optimizer import Optimizer as Optimizer
from .base_trainer import BaseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast

import lightning as L
import torch
from ..models.avclassify_model import BaseClassifierModel
from ..evaluation.complex import get_flops
from ..models import create_model
import copy
from ..utils.train_utils import get_checkpoint_files, get_newest_path
import os.path as osp
import yaml
from models.avclassify_model import MultiModalParallel
class UMTTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}, args = {}):
        super(UMTTrainer,self).__init__(fabric,**para_dict)
        self.scaling = method_dict['scaling']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']

        self.loaded_model = []
        loaded_model = {}
        temp_args = copy.deepcopy(args)
        # root_path = osp.dirname(osp.dirname(__file__))
        # with open(osp.join(root_path ,"configs", "encoder_config.yaml"), 'r') as f:
        #     encoder_settings = yaml.safe_load(f)
        if args.mode == "train_and_test":
            out_dir = '_'.join(temp_args.out_dir.split('/')[:-1])
            for modality in args.model['encoders'].keys():
                temp_args.model['encoders'] = {modality: args.model['encoders'][modality]}
                temp_args.model['modality_size'] = {modality: args.model['modality_size'][modality]}
                loaded_model[modality] = create_model(temp_args.model)
                out_dir = temp_args.out_dir.replace('UMTTrainer', 'unimodalTrainer_' + modality)
                out_dir = '/'.join(out_dir.split('/')[:-1])
                path = get_newest_path(out_dir)
                # loaded_model[modality].load_state_dict(torch.load(get_checkpoint_files(path)[0])['model'])
                loaded_model[modality].load_state_dict(torch.load(get_checkpoint_files(path)[0])['model'])
                loaded_model[modality] = MultiModalParallel(loaded_model[modality],device_ids=[0,1])
                loaded_model[modality] =loaded_model[modality].cuda()
                # loaded_model[modality] = torch.load(get_checkpoint_files(path)[0])['model']
                # print(type(loaded_model))
            self.loaded_model = loaded_model

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        self.fabric.call("on_train_epoch_start")
        all_modalitys = list(model.modalitys)
        all_modalitys.append('output')
        self.precision_calculator = self.PrecisionCalculatorType(model.n_classes, all_modalitys)
        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )
        # if self.current_epoch == 0:
            # for batch_idx, batch in enumerate(iterable):
            #     batch_sample = batch
            #     break
            # print(batch_sample.keys())
            # model_flops, _ =get_flops(model = model, input_sample = batch_sample)
            # self.FlopsMonitor.update(model_flops / len(batch_sample['label']) * len(train_loader), 'forward')
            # self.FlopsMonitor.report(logger = self.logger)
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer, 0)

                # optimizer step runs train step internally through closure
                optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx))
                self.fabric.call("on_before_zero_grad", optimizer)

                optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.precision_calculator.update(y_true = batch['label'].cpu(), y_pred = model.prediction)
            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            # if should_optim_step:
            #     self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            
            self._format_iterable(iterable, self._current_train_return, "train")
            
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)
        self._current_metrics = self.precision_calculator.compute_metrics()
        self.fabric.call("on_train_epoch_end")
    def training_step(self, model: BaseClassifierModel, batch, batch_idx):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        MSE = nn.MSELoss()
        label = batch['label']
        label = label.to(model.device)
        _ = model(batch= batch)
        out = model.encoder_result['output']        
        loss = criterion(out, label)
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            for modality in self.loaded_model.keys():
                with torch.no_grad():
                    self.loaded_model[modality](batch)
                out_unimodal = self.loaded_model[modality].encoder_result[modality]
                loss += self.scaling * MSE(out_unimodal, model.encoder_result[modality])

        loss.backward()
        return loss