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
from ..evaluation.complex import profile_flops

import lightning as L
import torch
import random

def conf_loss(conf, pred, conf_x, pred_x, label):
    #print(conf.shape, pred.shape, conf_x.shape, pred_x.shape, label.shape)
    # sign==1 => ( pred false || pred_x true)
    # sign == 0 => pred true and prex , 此时loss取0
    sign = (~((pred == label) & (pred_x != label))).long()  # trick 1
    #print(sign)
    return (max(0, torch.sub(conf_x, conf).sum())), sign.sum()

class CMLTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(CMLTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']

        self.lam = method_dict['lam']
        # self.modality = method_dict['modality']

    @profile_flops()
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

            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            # if should_optim_step:
            #     self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            
            self._format_iterable(iterable, self._current_train_return, "train")
            
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

        self.fabric.call("on_train_epoch_end")
    
    def training_step(self, model, batch, batch_idx, random_dict_ ):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        label = batch['label']
        
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
                    m[modality] = model.encoder_res[modality]
                m['out'] = model.encodr_res['output']
                # a, v, t, out = model(batch)
                Uni_res = model.Unimodality_Calculate()
                out_s = Uni_res['output']
                # out_a, out_v, out_t = model.AVTCalculate(a, v, t, out)
                # out_s = out
                random_dict = random_dict_.copy()
                for i in range(modality_num - 1):
                    removed_mm = random_dict.pop()
                    ### cuda out of memory
                    # _, _, _, out_s = model(batch, pad_audio = pad_audio, pad_visual = pad_visual, pad_text =pad_text)
                    # if removed_mm == 'audio':
                    #     pad_audio = True
                    # elif removed_mm == 'visual':
                    #     pad_visual = True
                    # else:
                    #     pad_text = True
                    # _, _, _, out_t = model(batch, pad_audio = pad_audio, pad_visual = pad_visual, pad_text =pad_text)
                    out_p = out_s - Uni_res[removed_mm] +model.fusion_module.fc_out.bias/3
                    # if removed_mm == 'audio':
                    #     out_p = out_s - out_a + model.fusion_module.fc_out.bias/3
                    # elif removed_mm == 'visual':
                    #     out_p = out_s - out_v + model.fusion_module.fc_out.bias/3
                    # else:
                    #     out_p = out_s - out_t + model.fusion_module.fc_out.bias/3
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
                    # if random_dict['audio']:
                    #     conf_a, _pred_a = torch.max(pred_a, dim=1)
                    #     loss_ac, count = conf_loss(conf, pred, conf_a, _pred_a, label)
                    #     # conf_loss_hit_a += count
                    #     loss += loss_a
                    #     _loss_c += loss_ac
                    # if random_dict["visual"]:
                    #     conf_v, _pred_v = torch.max(pred_v, dim=1)
                    #     loss_vc, count = conf_loss(conf, pred, conf_v, _pred_v, label)
                    #     # conf_loss_hit_v += count
                    #     loss += loss_v
                    #     _loss_c += loss_vc
                    # if random_dict['text']:
                    #     conf_t, _pred_t = torch.max(pred_t, dim=1)
                    #     loss_vc, count = conf_loss(conf, pred, conf_t, _pred_t, label)
                    #     # conf_loss_hit_v += count
                    #     loss += loss_v
                    #     _loss_c += loss_vc
                loss = (loss) / 3 +self.lam * _loss_c
            else:
                model(batch)
                for modality in modality_list:
                    m[modality] = model.encoder_res[modality]
                m['out'] = model.encoder_res['output']
                # a, v, t, out = model(batch)
                Uni_res = model.Unimodality_Calculate()
                out_s = Uni_res['output']
                # out_a, out_v, out_t = model.AVTCalculate(a, v, t, out)
                # print(a.shape, v.shape, model.head.weight.shape)

                ## our modality-wise normalization on weight and feature
            
                loss = criterion(m['out'], label)

        else:
            if self.modulation_starts <= self.current_epoch <= self.modulation_ends: ######
                pad_audio = False
                pad_visual = False
                pad_text = False
                loss_mm = 0
                model(batch)
                for modality in modality_list:
                    m[modality] = model.encoder_res[modality]
                m['out'] = model.encoder_res['output']
                Uni_res = model.Unimodality_Calculate()
                out_s = Uni_res['output']
                random_dict = random_dict_.copy()
                for i in range(modality_num - 1):
                    removed_mm = random_dict.pop()
                    ### cuda out of memory
                    # _, _, _, out_s = model(batch, pad_audio = pad_audio, pad_visual = pad_visual, pad_text =pad_text)
                    # if removed_mm == 'audio':
                    #     pad_audio = True
                    # elif removed_mm == 'visual':
                    #     pad_visual = True
                    # else:
                    #     pad_text = True
                    # _, _, _, out_t = model(batch, pad_audio = pad_audio, pad_visual = pad_visual, pad_text =pad_text)
                    out_p = out_s - Uni_res[removed_mm] +model.fusion_module.fc_out.bias/2
                    # if removed_mm == 'audio':
                    #     out_p = out_s - out_a + model.fusion_module.fc_out.bias/2
                    # elif removed_mm == 'visual':
                    #     out_p = out_s - out_v + model.fusion_module.fc_out.bias/2

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
                    m[modality] = model.encoder_res[modality]
                m['out'] = model.encoder_res['output']
                # out_a, out_v = model.AVCalculate(a, v, out)
            
                loss = criterion(m['out'], label)
        loss.backward()

        return loss