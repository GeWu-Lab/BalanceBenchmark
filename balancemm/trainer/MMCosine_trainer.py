from typing import Mapping
from lightning import LightningModule
from torch.optim.optimizer import Optimizer as Optimizer
from .base_trainer import BaseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
from ..evaluation.precisions import BatchMetricsCalculator
from ..models.avclassify_model import BaseClassifierModel

import lightning as L
from lightning_utilities import apply_to_collection
from itertools import combinations
from collections import defaultdict
from torch.utils.data.dataset import Dataset
import logging
from copy import deepcopy

    
class MMCosineTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(MMCosineTrainer,self).__init__(fabric,**para_dict)
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        
        self.scaling = method_dict['scaling']

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
    
        self.fabric.call("on_train_epoch_start")
        all_modalitys = list(model.modalitys)
        all_modalitys.append('output')
        self.precision_calculator = self.PrecisionCalculatorType(model.n_classes, all_modalitys)

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
    
    def training_step(self, model, batch, batch_idx):

        # TODO: make it simpler and easier to extend
        modality_list = model.modalitys
        key = list(modality_list)
        m = {}
        softmax = nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss()
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()
        # NCE = NCELoss(self.temperature, self.EPISILON)#
        # model.train()
        label = batch['label']
        label = label.to(model.device)
        # a, v, out = model(batch)
        model(batch)
        for modality in modality_list:
            m[modality] = model.encoder_result[modality]
        m['out'] = model.encoder_result['output']
        # nce_loss = NCE(m[key[0]], m[key[1]], label)
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            # size = model.fusion_module.fc_out.weight.size(1)
            unimodal_result = {}
            if len(key) == 2:
                unimodal_result[key[0]] = torch.mm(F.normalize(m[key[0]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :m[key[0]].size(1)], 0, 1), dim=0))
                unimodal_result[key[1]] = torch.mm(F.normalize(m[key[1]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1):], 0, 1), dim=0))
            if len(key) == 3:
                unimodal_result[key[0]] = torch.mm(F.normalize(m[key[0]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :m[key[0]].size(1)], 0, 1), dim=0))
                unimodal_result[key[1]] = torch.mm(F.normalize(m[key[1]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1):m[key[0]].size(1)+m[key[1]].size(1)], 0, 1), dim=0))
                unimodal_result[key[2]] = torch.mm(F.normalize(m[key[2]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1)+m[key[1]].size(1):], 0, 1), dim=0))
            for modality in modality_list:
                unimodal_result[modality] = unimodal_result[modality] * self.scaling
         
            unimodal_result['output'] = sum(unimodal_result[modality] for modality in key)
            model.unimodal_result = unimodal_result
            softmax =nn.Softmax(dim= 1)
            for modality in model.unimodal_result.keys():
                softmax_res = softmax(model.unimodal_result[modality])
                model.prediction[modality] = torch.argmax(softmax_res, dim = 1)
            loss = criterion(unimodal_result['output'],label)
            # loss = criterion(out, label) + self.lam * nce_loss
        else:
            loss = criterion(m['out'],label)
        loss.backward()

        return loss
    
    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
        limit_modalitys: list = ['ALL']
    ):

        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # # no validation but warning if val_loader was passed, but validation_step not implemented
        # if val_loader is not None and not is_overridden("validation_step", _unwrap_objects(model)):
        #     L.fabric.utilities.rank_zero_warn(
        #         "Your LightningModule does not have a validation_step implemented, "
        #         "but you passed a validation dataloder. Skipping Validation."
        #     )
        #     return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`
        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")
        
        if limit_modalitys == ["ALL"]:
            limit_modalitys = list(model.modalitys).copy()
        count = 0
        _acc = {}
        valid_loss = 0
        modalitys = list(model.modalitys)
        modalitys.append('output')
        MetricsCalculator = BatchMetricsCalculator(model.n_classes, modalitys)
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = self.validation_step(model, batch, batch_idx,limit_modalitys)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")
            # for modality in acc.keys():
            #     if modality not in _acc:
            #         _acc[modality] = 0
            #     _acc[modality] += sum(acc[modality])
            MetricsCalculator.update(y_true = batch['label'].cpu(), y_pred = model.prediction)
            valid_loss += out
            # count += len(batch['label'])
        valid_loss /= MetricsCalculator.total_samples
        Metrics_res = MetricsCalculator.compute_metrics()
        self._current_metrics = Metrics_res
        if Metrics_res['acc']['output'] > self.best_acc:
            self.should_save = True
            self.best_acc = Metrics_res['acc']['output']

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)
        return valid_loss, Metrics_res
    def validation_step(self, model, batch, batch_idx,limit_modalitys):
        
        padding = []
        for modality in model.modalitys:
            if modality not in limit_modalitys:
                padding.append(modality)
        model(batch, padding = padding)
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        label = label.to(model.device)
        key = list(model.modalitys)
        modality_list = model.modalitys
        m = {}
        for modality in modality_list:
            m[modality] = model.encoder_result[modality]
        m['out'] = model.encoder_result['output']
        if len(key) == 2:
            model.unimodal_result[key[0]] = torch.mm(F.normalize(m[key[0]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :m[key[0]].size(1)], 0, 1), dim=0))
            model.unimodal_result[key[1]] = torch.mm(F.normalize(m[key[1]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1):], 0, 1), dim=0))
        if len(key) == 3:
            model.unimodal_result[key[0]] = torch.mm(F.normalize(m[key[0]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :m[key[0]].size(1)], 0, 1), dim=0))
            model.unimodal_result[key[1]] = torch.mm(F.normalize(m[key[1]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1):m[key[0]].size(1)+m[key[1]].size(1)], 0, 1), dim=0))
            model.unimodal_result[key[2]] = torch.mm(F.normalize(m[key[2]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1)+m[key[1]].size(1):], 0, 1), dim=0))
        for modality in modality_list:
            model.unimodal_result[modality] = model.unimodal_result[modality] * self.scaling
        model.unimodal_result['output'] = sum(model.unimodal_result[modality] for modality in key)
        loss = F.cross_entropy(model.unimodal_result['output'], label)
        for modality in model.unimodal_result.keys():
            softmax_res = softmax(model.unimodal_result[modality])
            model.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        # for modality in self.unimodal_result.keys():
        #     acc_res[modality] = [0.0 for _ in range(n_classes)]
        #     pred_res[modality] = softmax(self.unimodal_result[modality])
        # for i in range(label.shape[0]):
        #     for modality in self.unimodal_result.keys():
        #         modality_pred = np.argmax(pred_res[modality][i].cpu().data.numpy())
        #         if np.asarray(label[i].cpu()) == modality_pred:
        #             acc_res[modality][label[i]] += 1.0
            
        #     num[label[i]] += 1.0
        return loss
    
   