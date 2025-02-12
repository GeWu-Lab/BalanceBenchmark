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
from lightning_utilities import apply_to_collection
import lightning as L
import torch
import random

def conf_loss(conf, pred, conf_x, pred_x, label):
    sign = (~((pred == label) & (pred_x != label))).long()  # trick 1
    #print(sign)
    return (max(0, torch.sub(conf_x, conf).sum())), sign.sum()

class CMLTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(CMLTrainer,self).__init__(fabric,**para_dict)

        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']

        self.lam = method_dict['lam']

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
        
        random_dict = list(model.modalitys)
        # if self.modality == 3:
        #     random_dict = ["audio", "visual", "text"]
        # else:
        #     random_dict = ['audio', "visual" ]
        random.shuffle(random_dict)
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
                optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx, random_dict_ = random_dict))
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
    
    def training_step(self, model, batch, batch_idx, random_dict_ ):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        label = batch['label']
        label = label.to(model.device)
        
        _loss_c = 0
        modality_num = len(model.modalitys)
        modality_list = model.modalitys
        key = list(modality_list)
        m = {}
        if modality_num == 3:
            if self.modulation_starts <= self.current_epoch <= self.modulation_ends: ######
                pad_audio = False
                pad_visual = False
                pad_text = False
                loss_mm = 0
                model(batch)
                for modality in modality_list:
                    m[modality] = model.encoder_result[modality]
                m['out'] = model.encoder_result['output']
                # a, v, t, out = model(batch)
                unimodal_result = model.Unimodality_Calculate()
                out_s = unimodal_result['output']
                # out_a, out_v, out_t = model.AVTCalculate(a, v, t, out)
                # out_s = out
                random_dict = random_dict_.copy()
                for i in range(modality_num - 1):
                    removed_mm = random_dict.pop()
                   
                    out_p = out_s - unimodal_result[removed_mm] +model.fusion_module.fc_out.bias/3
             
                    prediction_s = softmax(out_s)
                    conf_s, pred_s = torch.max(prediction_s, dim=1)

                    prediction_p = softmax(out_p)
                    conf_p, pred_p = torch.max(prediction_p, dim=1)
                    
                    if i ==0 : loss = criterion(out_s, label)
                    
                    loss_p = criterion(out_p, label)
                    loss_pc ,_ = conf_loss(conf_s, pred_s, conf_p, pred_p, label)
                    loss = loss + loss_p
                    _loss_c = _loss_c + loss_pc

                    out_s = out_p
                    
                loss = (loss) / 3 +self.lam * _loss_c
            else:
                model(batch)
                for modality in modality_list:
                    m[modality] = model.encoder_result[modality]
                m['out'] = model.encoder_result['output']
                # a, v, t, out = model(batch)
                unimodal_result = model.Unimodality_Calculate()
                out_s = unimodal_result['output']
                
            
                loss = criterion(m['out'], label)

        else:
            if self.modulation_starts <= self.current_epoch <= self.modulation_ends: ######
                pad_audio = False
                pad_visual = False
                pad_text = False
                loss_mm = 0
                model(batch)
                for modality in modality_list:
                    m[modality] = model.encoder_result[modality]
                m['out'] = model.encoder_result['output']
                unimodal_result = model.Unimodality_Calculate()
                out_s = unimodal_result['output']
                random_dict = random_dict_.copy()
                for i in range(modality_num - 1):
                    removed_mm = random_dict.pop()
                    
                    out_p = out_s - unimodal_result[removed_mm] +model.fusion_module.fc_out.bias/2

                    prediction_s = softmax(out_s)
                    conf_s, pred_s = torch.max(prediction_s, dim=1)

                    prediction_p = softmax(out_p)
                    conf_p, pred_p = torch.max(prediction_p, dim=1)
                    
                    if i ==0 : loss = criterion(out_s, label)
                    
                    loss_p = criterion(out_p, label)
                    loss_pc ,_ = conf_loss(conf_s, pred_s, conf_p, pred_p, label)
                    loss += loss_p
                    _loss_c += loss_pc

                    out_s = out_p
                loss = (loss) / 2 +self.lam * _loss_c
            else:
                model(batch)
                for modality in modality_list:
                    m[modality] = model.encoder_result[modality]
                m['out'] = model.encoder_result['output']
                # out_a, out_v = model.AVCalculate(a, v, out)
            
                loss = criterion(m['out'], label)
        loss.backward()

        # # avoid gradients in stored/accumulated values -> prevents potential OOM
        # self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())


        return loss