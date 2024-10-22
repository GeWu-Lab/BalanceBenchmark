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
import numpy as np
from ..models.avclassify_model import BaseClassifier_GreedyModel
class GreedyTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(GreedyTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        
        self.modality = method_dict['modality']
        self.window_size = method_dict['window_size']
        self.M_bypass_modal_0 = 0
        self.M_bypass_modal_1 = 0
        self.M_main_modal_0 = 0
        self.M_main_modal_1 = 0
        self.curation_mode = False
        self.caring_modality = 0
        self.curation_step = self.window_size
        self.speed = 0
    
    def train_loop(
        self,
        model: BaseClassifier_GreedyModel,
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
        self.PrecisionCalculator = self.PrecisionCalculatorType(model.n_classes, all_modalitys)
        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )
        
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
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)
                if not self.curation_mode:
                    self.speed = self.compute_learning_speed(model)
                    if abs(self.speed) > self.alpha:
                        biased_direction=np.sign(self.speed)
                        self.curation_mode = True
                        self.curation_step = 0

                        if biased_direction==-1: #BDR0<BDR1
                            self.caring_modality = 1
                        elif biased_direction==1: #BDR0>BDR1
                            self.caring_modality = 0 
                    else:
                        self.curation_mode = False 
                        self.caring_modality = 0 
                else:
                    self.curation_step +=1
                    if self.curation_step==self.window_size:
                        self.curation_mode=False
                optimizer.step()
                self.fabric.call("on_before_zero_grad", optimizer)

                optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.PrecisionCalculator.update(y_true = batch['label'].cpu(), y_pred = model.pridiction)
            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            # if should_optim_step:
            #     self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            
            self._format_iterable(iterable, self._current_train_return, "train")
            
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

        self._current_metrics = self.PrecisionCalculator.compute_metrics()
        self.fabric.call("on_train_epoch_end")
    
    def training_step(self, model: BaseClassifier_GreedyModel, batch, batch_idx):

        # TODO: make it simpler and easier to extend
        modality_list = model.modalitys
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            model(batch,self.curation_mode,self.caring_modality)
        else:
            model(batch,self.curation_mode,self.caring_modality)
        model.Unimodality_Calculate()
      
        label = batch['label']
        label = label.to(model.device)
        # print(a.shape, v.shape, model.head.weight.shape)

        ## our modality-wise normalization on weight and feature
        out = model.encoder_res['output']
        loss = criterion(out, label) 
        loss.backward()
        return loss
    
    def compute_learning_speed(self,model:BaseClassifier_GreedyModel):
        modality_list = model.modalitys
        wn_main, wn_bypass = [0]*len(modality_list), [0]*len(modality_list)
        gn_main, gn_bypass = [0]*len(modality_list), [0]*len(modality_list)
        for name, parameter in model.named_parameters():
            wn = (parameter ** 2).sum().item()
            gn = (parameter.grad.data ** 2).sum().item()#(grad ** 2).sum().item()
            if 'mmtm_layers' in name:
                shared=True
                for ind, modal in enumerate(modality_list):
                    if modal in name: 
                        wn_bypass[ind]+=wn
                        gn_bypass[ind]+=gn
                        shared = False
                if shared:
                    for ind, modal in enumerate(modality_list):
                        wn_bypass[ind]+=wn
                        gn_bypass[ind]+=gn

            else:
                for ind, modal in enumerate(modality_list):
                    if modal in name: 
                        wn_main[ind]+=wn
                        gn_main[ind]+=gn

        self.M_bypass_modal_0 += gn_bypass[0]/wn_bypass[0]
        self.M_bypass_modal_1 += gn_bypass[1]/wn_bypass[1]
        self.M_main_modal_0 += gn_main[0]/wn_main[0]
        self.M_main_modal_1 += gn_main[1]/wn_main[1]

        BDR_0 = np.log10(self.M_bypass_modal_0/self.M_main_modal_0)
        BDR_1 = np.log10(self.M_bypass_modal_1/self.M_main_modal_1)

        return BDR_0 - BDR_1