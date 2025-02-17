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


class MBSDTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(MBSDTrainer,self).__init__(fabric,**para_dict)
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']

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
    
    def training_step(self, model, batch, batch_idx , dependent_modality : str = 'none'):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        modality_list = model.modalitys
        key = list(modality_list)
        m = {}
        loss = {}
        loss_modality = {}
        prediction = {}
        y_pred = {}
        model(batch)
        for modality in modality_list:
            m[modality] = model.encoder_result[modality]
        m['out'] = model.encoder_result['output']
        model.Unimodality_Calculate()    
        # out_a, out_v = model.AVCalculate(a, v, out)
        label = batch['label']
        device = model.device
        # print(a.shape, v.shape, model.head.weight.shape)

        ## our modality-wise normalization on weight and feature
    
        loss['out'] = criterion(m['out'], label)
        for modality in modality_list:
            loss_modality[modality] = criterion(model.unimodal_result[modality], label)
        # loss_v = criterion(unimodal_result[], label)
        # loss_a = criterion(out_a, label)

        for modality in modality_list:
            prediction[modality] = softmax(model.unimodal_result[modality])
        # prediction_a = softmax(out_a)
        # prediction_v = softmax(out_v)
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            if len(modality_list) == 2:
                
                loss_RS = 1/model.unimodal_result[key[0]].shape[1] * torch.sum((model.unimodal_result[key[0]] - model.unimodal_result[key[1]])**2, dim = 1)

                w = torch.tensor([0.0 for _ in range(len(m['out']))])
                w = w.to(device)
                for modality in modality_list:
                    y_pred[modality] = prediction[modality]
                    y_pred[modality] = y_pred[modality].argmax(dim=-1)
                # y_pred_a = prediction_a
                # y_pred_a = y_pred_a.argmax(dim = -1)
                # y_pred_v = prediction_v
                # y_pred_v = y_pred_v.argmax(dim = -1)
                ps = torch.tensor([0.0 for _ in range(len(m['out']))])
                ps = ps.to(device)
                pw = torch.tensor([0.0 for _ in range(len(m['out']))])
                pw = pw.to(device)
                for i in range(len(m['out'])):
                    if y_pred[key[0]][i] == label[i] or y_pred[key[1]][i] == label[i]:
                        w[i] = max(prediction[key[0]][i][label[i]], prediction[key[1]][i][label[i]]) -  min(prediction[key[0]][i][label[i]], prediction[key[1]][i][label[i]])
                    ps[i] = max(prediction[key[0]][i][label[i]], prediction[key[1]][i][label[i]])
                    pw[i] = min(prediction[key[0]][i][label[i]], prediction[key[1]][i][label[i]])

                loss_KL = F.kl_div(ps, pw, reduction = 'none')
                w = w.reshape(1,-1)
                loss_KL = loss_KL.reshape(-1,1)
                loss_KL = torch.mm(w, loss_KL) / len(m['out'])
                loss_RS = loss_RS.reshape(-1,1)
                loss_RS = torch.mm(w, loss_RS) / len(m['out'])
                total_loss = loss['out'] + loss_modality[key[0]] + loss_modality[key[1]] + loss_RS.squeeze() + loss_KL.squeeze() ## erase the dim of 1
            else:

                w1 = torch.tensor([0.0 for _ in range(len(m['out']))])
                w1 = w1.to(device)
                w2 = torch.tensor([0.0 for _ in range(len(m['out']))])
                w2 = w2.to(device)
                w3 = torch.tensor([0.0 for _ in range(len(m['out']))])
                w3 = w3.to(device)
                ps1 = torch.tensor([0.0 for _ in range(len(m['out']))])
                ps2 = torch.tensor([0.0 for _ in range(len(m['out']))])
                ps3 = torch.tensor([0.0 for _ in range(len(m['out']))])
                ps1 = ps1.to(device)
                ps2 = ps2.to(device)
                ps3 = ps3.to(device)
                pw1 = torch.tensor([0.0 for _ in range(len(m['out']))])
                pw2 = torch.tensor([0.0 for _ in range(len(m['out']))])
                pw3 = torch.tensor([0.0 for _ in range(len(m['out']))])
                pw1 = pw1.to(device)
                pw2 = pw2.to(device)
                pw3 = pw3.to(device)
                for modality in modality_list:
                    y_pred[modality] = prediction[modality]
                    y_pred[modality] = y_pred[modality].argmax(dim=-1)
                    
                for i in range(len(m['out'])):
                    if y_pred[key[0]][i] == label[i] or y_pred[key[1]][i] == label[i]:
                        w1[i] = max(prediction[key[0]][i][label[i]], prediction[key[1]][i][label[i]]) -  min(prediction[key[0]][i][label[i]], prediction[key[1]][i][label[i]])
                    if y_pred[key[0]][i] == label[i] or y_pred[key[2]][i] == label[i]:
                        w2[i] = max(prediction[key[0]][i][label[i]], prediction[key[2]][i][label[i]]) -  min(prediction[key[0]][i][label[i]], prediction[key[2]][i][label[i]])
                    if y_pred[key[1]][i] == label[i] or y_pred[key[2]][i] == label[i]:
                        w3[i] = max(prediction[key[1]][i][label[i]], prediction[key[2]][i][label[i]]) -  min(prediction[key[1]][i][label[i]], prediction[key[2]][i][label[i]])
                    ps1[i] = max(prediction[key[0]][i][label[i]], prediction[key[1]][i][label[i]])
                    pw1[i] = min(prediction[key[0]][i][label[i]], prediction[key[1]][i][label[i]])
                    ps2[i] = max(prediction[key[0]][i][label[i]], prediction[key[2]][i][label[i]])
                    pw2[i] = min(prediction[key[0]][i][label[i]], prediction[key[2]][i][label[i]])
                    ps3[i] = max(prediction[key[1]][i][label[i]], prediction[key[2]][i][label[i]])
                    pw3[i] = min(prediction[key[1]][i][label[i]], prediction[key[2]][i][label[i]])
                loss_RS1 = 1/model.unimodal_result[key[0]].shape[1] * torch.sum((prediction[key[0]]-prediction[key[1]])**2,dim=1)
                loss_RS2 = 1/model.unimodal_result[key[0]].shape[1] * torch.sum((prediction[key[0]]-prediction[key[2]])**2,dim=1)
                loss_RS3 = 1/model.unimodal_result[key[0]].shape[1] * torch.sum((prediction[key[1]]-prediction[key[2]])**2,dim=1)
            
                loss_KL1 = F.kl_div(ps1, pw1, reduction = 'none')
                loss_KL2 = F.kl_div(ps2, pw2, reduction = 'none')
                loss_KL3 = F.kl_div(ps3, pw3, reduction = 'none')
                
                w1 = w1.reshape(1,-1)
                w2 = w2.reshape(1,-1)
                w3 = w3.reshape(1,-1)
                loss_KL1 = loss_KL1.reshape(-1,1)
                loss_KL1 = torch.mm(w1, loss_KL1) / len(m['out'])
                loss_KL2 = loss_KL2.reshape(-1,1)
                loss_KL2 = torch.mm(w2, loss_KL2) / len(m['out'])
                loss_KL3 = loss_KL3.reshape(-1,1)
                loss_KL3 = torch.mm(w3, loss_KL3) / len(m['out'])
                loss_KL = (loss_KL1 + loss_KL2 + loss_KL3) / 3
                
                loss_RS1 = loss_RS1.reshape(-1,1)
                loss_RS2 = loss_RS2.reshape(-1,1)
                loss_RS3 = loss_RS3.reshape(-1,1)
                loss_RS1 = torch.mm(w1, loss_RS1) / len(m['out'])
                loss_RS2 = torch.mm(w2, loss_RS2) / len(m['out'])
                loss_RS3 = torch.mm(w3, loss_RS3) / len(m['out'])
                loss_RS = (loss_RS1 + loss_RS2 + loss_RS3) / 3
                
                total_loss = loss['out'] + loss_modality[key[0]] + loss_modality[key[1]] + loss_modality[key[2]] + loss_KL.squeeze() + loss_RS.squeeze()## erase the dim of 1
                
        else:
        
            total_loss = loss['out']
        total_loss.backward()

        return total_loss