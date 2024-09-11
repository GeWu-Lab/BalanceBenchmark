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
import torch
from lightning_utilities import apply_to_collection

class NCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1,EPISILON = 1e-5):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)
        self.EPISILON = EPISILON

    def where(self, cond, x_1, x_2):
        cond = cond.type(torch.float32)
        return (cond * x_1) + ((1 - cond) * x_2)

    def forward(self, f1, f2, targets):
        ### cuda implementation
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        ## set distances of the same label to zeros
        mask = targets.unsqueeze(1) - targets
        self_mask = (torch.zeros_like(mask) != mask).float()  ### where the negative samples are labeled as 1
        dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)

        ## convert l2 distance to cos distance
        cos = 1 - 0.5 * dist

        ## convert cos distance to exponential space
        pred_softmax = self.softmax(cos / self.temperature)  ### convert to multi-class prediction scores

        log_pos_softmax = - torch.log(pred_softmax + self.EPISILON) * (1 - self_mask.float())
        log_neg_softmax = - torch.log(1 - pred_softmax + self.EPISILON) * self_mask.float()
        log_softmax = log_pos_softmax.sum(1) / (1 - self_mask).sum(1).float() + log_neg_softmax.sum(1) / self_mask.sum(
            1).float()
        loss = log_softmax

        return loss.mean()
    
class MMCosineTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(MMCosineTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
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
        NCE = NCELoss(self.temperature, self.EPISILON)#
        model.train()
        label = batch['label']
        label = label.to(model.device)
        a, v, out = model(batch)
        nce_loss = NCE(a, v, label)
        size = model.fusion_module.fc_out.weight.size(1)
        out_a = torch.mm(F.normalize(a, dim=1),
                            F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :size//2], 0, 1),
                                        dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
        out_v = torch.mm(F.normalize(v, dim=1),
                            F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, size//2:], 0, 1),
                                        dim=0))
        out_a = out_a * self.scaling #
        out_v = out_v * self.scaling
        out = out_a + out_v 
        loss = criterion(out, label) + self.lam * nce_loss
        loss.backward()

        return loss
    
    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
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

        count = 0
        _acc = 0
        _acc_a = 0
        _acc_v = 0
        _acc_t = 0
        valid_loss = 0
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out, acc, acc_a, acc_v, acc_t = self.validation_step(model,batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")

            
            _acc += acc
            _acc_a += acc_a
            _acc_v += acc_v
            _acc_t += acc_t
            valid_loss += out
            count += len(batch['label'])
        valid_loss /= count
        _acc /= count
        _acc_a /= count
        _acc_v /= count
        _acc_t /= count
        if _acc > self.best_acc:
            self.should_save = True
            self.best_acc = _acc

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)
        return valid_loss, _acc, _acc_a, _acc_v, _acc_t
    def validation_step(self, model, batch, batch_idx):

        # TODO: make it simpler and easier to extend
        softmax = nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss()
        NCE = NCELoss(self.temperature, self.EPISILON)#
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]
        model.train()
        label = batch['label']
        label = label.to(model.device)
        a, v, out = model(batch)
        nce_loss = NCE(a, v, label)
        n_classes = self.n_classes
        size = model.fusion_module.fc_out.weight.size(1)
        out_a = torch.mm(F.normalize(a, dim=1),
                            F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, :size//2], 0, 1),
                                        dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
        out_v = torch.mm(F.normalize(v, dim=1),
                            F.normalize(torch.transpose(model.fusion_module.fc_out.weight[:, size//2:], 0, 1),
                                        dim=0))
        out_a = out_a * self.scaling #
        out_v = out_v * self.scaling
        out = out_a + out_v 
        loss = criterion(out, label) + self.lam * nce_loss
        loss.backward()


        prediction = softmax(out)
        pred_v = softmax(out_v)
        pred_a = softmax(out_a)
        

        for i in range(label.shape[0]):

            ma = np.argmax(prediction[i].cpu().data.numpy())
            v = np.argmax(pred_v[i].cpu().data.numpy())
            a = np.argmax(pred_a[i].cpu().data.numpy())
            num[label[i]] += 1.0

            #pdb.set_trace()
            if np.asarray(label[i].cpu()) == ma:
                acc[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == v:
                acc_v[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == a:
                acc_a[label[i]] += 1.0

        return loss, sum(acc), sum(acc_a), sum(acc_v), sum(acc_t)