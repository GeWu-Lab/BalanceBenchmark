from typing import Mapping
from torch.optim.optimizer import Optimizer as Optimizer
from .base_trainer import BaseTrainer

import torch
import torch.nn as nn
from balancemm.models.avclassify_model import BaseClassifierModel
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
        # self.modality = method_dict['modality']

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
                # torch.cuda.empty_cache()
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
    
    def training_step(self, model : BaseClassifierModel, batch, batch_idx):

        # TODO: make it simpler and easier to extend
        softmax = nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss()
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()
        label = batch['label']
        label = label.to(model.device)
        model(batch)
        Uni_res = model.Unimodality_Calculate()
        loss = criterion(Uni_res['output'], label)
        loss.backward()
        modality_list = model.modalitys

        # Modulation starts here !
        modality_nums = len(modality_list) 
        scores = {}
        ratios = {}
        coeffs = {}
        minscore = float('inf')
        #Calculate the scores

        for modality in modality_list:
            if modality_nums == 2:
                score_modality = sum([softmax(Uni_res[modality])[i][label[i]] for i in range(Uni_res['output'].size(0))])
            elif modality_nums == 3:
                score_modality = sum([softmax(torch.cos(Uni_res[modality]))[i][label[i]] if label[i] == torch.argmax(Uni_res[modality][i]) else 0 for i in range(Uni_res['output'].size(0))])
            else:
                raise("Wrong number of modalitys for OGM, it should be 2 or 3, but given {:0}".format(modality_nums))
            scores[modality] = score_modality.detach().clone()
            minscore = min(score_modality, minscore)
        ##Calculate the ratios
        for modality in modality_list:
            ratios[modality] = scores[modality].detach().clone()
            if modality_nums == 2:
                for modality_another in modality_list:
                    if modality_another == modality: 
                        continue
                    ## 如果没有1e-3会显存爆炸
                    ratios[modality] /= (scores[modality_another]+ 1e-5)
            if modality_nums == 3:
                ratios[modality] /= (minscore + 1e-5)
        #Calculate the coeffects
        for modality in modality_list:
            if ratios[modality] > 1 :
                coeffs[modality] = max(1 - tanh(self.alpha * relu(ratios[modality])),0)
            else:
                coeffs[modality] = 1

        if self.modulation_starts <= self.current_epoch <= self.modulation_ends: # bug fixed
            for name, parms in model.named_parameters():
                layer = str(name).split('.')[1]
                for modality in modality_list:

                    if modality in layer and len(parms.grad.size()) != 1: ##Don't change the grad of bias for layer
                        if self.method == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeffs[modality] + \
                                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif self.method == 'OGM':
                            parms.grad *= coeffs[modality] 
        else:
            pass


        # model.Uni_res.clear()
        # model.encoder_res.clear()
        # scores.clear()
        # ratios.clear()
        # coeffs.clear()
        # batch.clear()

        return loss