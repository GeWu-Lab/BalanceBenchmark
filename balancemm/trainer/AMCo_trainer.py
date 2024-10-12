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
from ..models.avclassify_model import BaseClassifierModel
class AMCoTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(AMCoTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        
        self.sigma = method_dict['sigma']
        self.U = method_dict['U']
        self.eps = method_dict['eps']
        self.modality = method_dict['modality']
    
    def train_loop(
        self,
        model: BaseClassifierModel,
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

        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )
        
        dependent_modality = {}
        for modality in model.modalitys:
            dependent_modality[modality] = False
        l_t = 0
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # prepare the mask
            pt = np.sin(np.pi/2*(min(self.eps,l_t)/self.eps))
            N = int(pt * model.n_classes)
            mask_t = np.ones(model.n_classes-N)
            mask_t = np.pad(mask_t,(0,N))
            np.random.shuffle(mask_t)
            mask_t = torch.from_numpy(mask_t)
            mask_t = mask_t.to(model.device)
            mask_t = mask_t.float()
            l_t += self.current_epoch/10

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer, 0)

                # optimizer step runs train step internally through closure
                loss, dependent_modality = self.training_step( model=model, batch=batch, 
                                                              batch_idx=batch_idx, mask= mask_t, 
                                                              dependent_modality= dependent_modality,
                                                              pt = pt)
                optimizer.step()
                self.fabric.call("on_before_zero_grad", optimizer)

                optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            # if should_optim_step:
            #     self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            
            self._format_iterable(iterable, self._current_train_return, "train")
            
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

        self.fabric.call("on_train_epoch_end")
    
    def training_step(self, model: BaseClassifierModel, batch, batch_idx, dependent_modality, mask ,pt):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            model(batch,dependent_modality = dependent_modality, mask = mask,\
                                            pt = pt)
        else:
            model(batch)
        model.Unimodality_Calculate(mask, dependent_modality)
      
        label = batch['label']
        label = label.to(model.device)
        # print(a.shape, v.shape, model.head.weight.shape)

        ## our modality-wise normalization on weight and feature
        out = model.Uni_res['output']
        loss = criterion(out, label) 
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            loss.backward(retain_graph = True)
            for modality in model.modalitys:      
                loss_uni = criterion(model.Uni_res[modality],label)
                loss_uni.backward()
            out_combine = torch.cat([value for key,value in model.Uni_res.items() if key != 'output'],1)
            sft_out = softmax(out_combine)
            now_dim = 0
            for modality in model.modalitys:
                if now_dim < sft_out.shape[1] - model.n_classes:
                    sft_uni = torch.sum(sft_out[:, now_dim: now_dim + model.n_classes])/(len(label))
                else:
                    sft_uni = torch.sum(sft_out[:, now_dim: ])/(len(label))
                dependent_modality[modality] = sft_uni > self.sigma
                now_dim += model.n_classes
        else:
            loss.backward()

        return loss, dependent_modality

class AMCoTrainer_2(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(AMCoTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        
        self.sigma = method_dict['sigma']
        self.U = method_dict['U']
        self.eps = method_dict['eps']
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
        self.PrecisionCalculator = self.PrecisionCalculatorType(model.n_classes, all_modalitys)
        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )
        
        dependent_modality = {"audio": False, "visual": False, "text": False}
        l_t = 0

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # prepare the mask
            pt = np.sin(np.pi/2*(min(self.eps,l_t)/self.eps))
            N = int(pt * self.U)
            mask_t = np.ones(self.U-N)
            mask_t = np.pad(mask_t,(0,N))
            np.random.shuffle(mask_t)
            mask_t = torch.from_numpy(mask_t)
            mask_t = mask_t.to(next(model.parameters()).device)
            l_t += self.current_epoch/10

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer, 0)

                # optimizer step runs train step internally through closure
                loss, dependent_modality = self.training_step( model=model, batch=batch, 
                                                              batch_idx=batch_idx, mask= mask_t, 
                                                              dependent_modality= dependent_modality,
                                                              pt = pt)
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
    
    def training_step(self, model, batch, batch_idx, dependent_modality, mask ,pt):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            a, v, out = model(batch,dependent_modality = dependent_modality, mask = mask,\
                                         pt = pt)
        else:
            a, v, out = model(batch)
        out_a, out_v = model.AVCalculate(a, v, out)
        label = batch['label']
        label = label.to(model.device)
        # print(a.shape, v.shape, model.head.weight.shape)

        ## our modality-wise normalization on weight and feature
    
        loss = criterion(out, label) 
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)
            loss.backward(retain_graph=True)
            loss_v.backward()
            loss_a.backward()


            out_combine = torch.cat((out_a, out_v, out_t),1)
            sft_out = softmax(out_combine)
            sft_oa = torch.sum(sft_out[:, 0: model.n_classes])/(len(label))
            sft_ov = torch.sum(sft_out[:, model.n_classes:2* model.n_classes])/(len(label))
            # print(sft_oa,sft_ov,sft_out.size())
            if(sft_oa>=self.sigma):
                dependent_modality['audio'] = True
            elif(sft_ov>=self.sigma):
                dependent_modality['visual'] = True
            elif(sft_ot>=self.sigma):
                dependent_modality['text'] = True
        else: 
            loss.backward()
        
        


        return loss, dependent_modality