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

import lightning as L
from lightning_utilities import apply_to_collection


# class NCELoss(torch.nn.Module):
#     def __init__(self, temperature=0.1,EPISILON = 1e-5):
#         super(NCELoss, self).__init__()
#         self.temperature = temperature
#         self.softmax = nn.Softmax(dim=1)
#         self.EPISILON = EPISILON

#     def where(self, cond, x_1, x_2):
#         cond = cond.type(torch.float32)
#         return (cond * x_1) + ((1 - cond) * x_2)

#     def forward(self, f1, f2, targets):
#         ### cuda implementation
#         f1 = F.normalize(f1, dim=1)
#         f2 = F.normalize(f2, dim=1)

#         ## set distances of the same label to zeros
#         mask = targets.unsqueeze(1) - targets
#         self_mask = (torch.zeros_like(mask) != mask).float()  ### where the negative samples are labeled as 1
#         dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)

#         ## convert l2 distance to cos distance
#         cos = 1 - 0.5 * dist

#         ## convert cos distance to exponential space
#         pred_softmax = self.softmax(cos / self.temperature)  ### convert to multi-class prediction scores

#         log_pos_softmax = - torch.log(pred_softmax + self.EPISILON) * (1 - self_mask.float())
#         log_neg_softmax = - torch.log(1 - pred_softmax + self.EPISILON) * self_mask.float()
#         log_softmax = log_pos_softmax.sum(1) / (1 - self_mask).sum(1).float() + log_neg_softmax.sum(1) / self_mask.sum(
#             1).float()
#         loss = log_softmax

#         return loss.mean()
    
class MMCosineTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(MMCosineTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        
        self.scaling = method_dict['scaling']

    # @profile_flops()
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
            m[modality] = model.encoder_res[modality]
        m['out'] = model.encoder_res['output']
        # nce_loss = NCE(m[key[0]], m[key[1]], label)
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            # size = model.fusion_module.fc_out.weight.size(1)
            Uni_res = {}
            if len(key) == 2:
                Uni_res[key[0]] = torch.mm(F.normalize(m[key[0]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :m[key[0]].size(1)], 0, 1), dim=0))
                Uni_res[key[1]] = torch.mm(F.normalize(m[key[1]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1):], 0, 1), dim=0))
            if len(key) == 3:
                Uni_res[key[0]] = torch.mm(F.normalize(m[key[0]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :m[key[0]].size(1)], 0, 1), dim=0))
                Uni_res[key[1]] = torch.mm(F.normalize(m[key[1]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1):m[key[0]].size(1)+m[key[1]].size(1)], 0, 1), dim=0))
                Uni_res[key[2]] = torch.mm(F.normalize(m[key[2]], dim=1), F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, m[key[0]].size(1)+m[key[1]].size(1):], 0, 1), dim=0))
            for modality in modality_list:
                Uni_res[modality] = Uni_res[modality] * self.scaling
            # out_a = torch.mm(F.normalize(a, dim=1),
            #                     F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :size//2], 0, 1),
            #                                 dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            # out_v = torch.mm(F.normalize(v, dim=1),
            #                     F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, size//2:], 0, 1),
            #                                 dim=0))
            # out_a = out_a * self.scaling #
            # out_v = out_v * self.scaling
            # out = out_a + out_v 
            out = sum(Uni_res[modality] for modality in key)
            loss = criterion(out,label)
            # loss = criterion(out, label) + self.lam * nce_loss
        else:
            loss = criterion(m['out'],label)
        loss.backward()

        return loss
    