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


class OGMTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(OGMTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.modality = method_dict['modality']

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

            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            # if should_optim_step:
            #     self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            
            self._format_iterable(iterable, self._current_train_return, "train")
            
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

        self.fabric.call("on_train_epoch_end")
    
    def training_step(self, model, batch, batch_idx):

        # TODO: make it simpler and easier to extend
        softmax = nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss()
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()
        label = batch['label']
        label = label.to(model.device)
        if self.modality == 3:
            a, v, t, out = model(batch)
            out_a, out_v, out_t = model.AVTCalculate(a, v, t, out)
        a, v, out = model(batch)
        out_a, out_v = model.AVCalculate(a, v, out)
        loss = criterion(out, label)
        loss.backward()


        # Modulation starts here !

        score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
        score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
        if self.modality == 3:
            size = model.fusion_module.fc_out.weight.size(1)
            dot_a = torch.mm(F.normalize(a, dim=1),
                            F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :size//3], 0, 1),
                                        dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            dot_v = torch.mm(F.normalize(v, dim=1),
                            F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, size//3:size//3*2], 0, 1),
                                        dim=0))
            dot_t = torch.mm(F.normalize(t, dim=1),
                            F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, size//3*2:], 0, 1),
                                        dim=0))
            score_a = sum([softmax(torch.cos(dot_t))[i][label[i]]] if label[i] == torch.argmax(out_t[i]) else 0 for i in range(len(out)) )
            score_v = sum([softmax(torch.cos(dot_t))[i][label[i]]] if label[i] == torch.argmax(out_t[i]) else 0 for i in range(len(out)) )
            score_t = sum([softmax(torch.cos(dot_t))[i][label[i]]] if label[i] == torch.argmax(out_t[i]) else 0 for i in range(len(out)) )
            minscore = min([score_v, score_a, score_t])
            ratio_a = score_a / minscore
            ratio_v = score_v / minscore
            ratio_t = score_t / minscore
            coeff_a = 1
            coeff_v = 1
            coeff_t = 1
            if ratio_a > 1:
                coeff_a = 1 - tanh(self.alpha * relu(ratio_a))
            if ratio_v > 1:
                coeff_v = 1 - tanh(self.alpha * relu(ratio_v))
            if ratio_t > 1:
                coeff_t = 1 - tanh(self.alpha * relu(ratio_t))
        else:
            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """
            if ratio_v > 1:
                coeff_v = 1 - tanh(self.alpha * relu(ratio_v))
                coeff_a = 1
            else:
                coeff_a = 1 - tanh(self.alpha * relu(ratio_a))
                coeff_v = 1

        if self.modulation_starts <= self.current_epoch <= self.modulation_ends: # bug fixed
            for name, parms in model.named_parameters():
                layer = str(name).split('.')[0]
                if 'audio' in layer and len(parms.grad.size()) != 1: # according to your model 
                    if self.method == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_a + \
                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif self.method == 'OGM':
                        parms.grad *= coeff_a

                if 'visual' in layer and len(parms.grad.size()) != 1:
                    if self.method == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_v + \
                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif self.method == 'OGM':
                        parms.grad *= coeff_v
                if self.modality == 3:
                    if 'text' in layer and len(parms.grad.size()) != 1:
                        if self.method == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_t + \
                                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif self.method == 'OGM':
                            parms.grad *= coeff_v
        else:
            pass


    

        return loss