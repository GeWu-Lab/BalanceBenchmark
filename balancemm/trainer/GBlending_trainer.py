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

class GBlendingTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(GBlendingTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        
        self.super_epoch = method_dict['super_epoch']
        self.modality = method_dict['modality']

    def fit(
        self,
        model,
        temp_model,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: torch.optim.lr_scheduler,
        logger,
        ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """
        print(self.max_epochs)
        self.fabric.launch()

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)
        # optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
        # assert optimizer is not None
        # model, optimizer = self.fabric.setup(model, optimizer)

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True
        weight_a = 0
        weight_v = 0
        weight_t = 0
        weight_avt = 0
        while not self.should_stop:
            if self.current_epoch % self.super_epoch == 0:
                weight_a, weight_v, weight_t, weight_avt = self.super_epoch_origin(model, temp_model, self.limit_train_batches, train_loader, val_loader, optimizer,logger)
            self.train_loop(
                model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg,
                weight_a = weight_a, weight_v = weight_v, weight_t = weight_t, weight_avt = weight_avt
            )
            logger.info("epoch: {:0}  ".format(self.current_epoch))

            if self.should_validate:
                valid_loss, valid_acc, valid_acc_a,valid_acc_v, valid_acc_t=self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)
                
                logger.info(" valid_loss: {:0}, valid_acc: {:1}, acc_a: {:2}, acc_v: {:3}, acc_t: {:4}"
                        .format( valid_loss, valid_acc, valid_acc_a, valid_acc_v, valid_acc_t))
                for handler in logger.handlers:
                    handler.flush()
            # self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)
            scheduler_cfg.step()

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True
            
            if self.should_save:
                self.save(state)
                self.should_save = False

        # reset for next fit call
        self.should_stop = False
    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
        weight_a = 1,
        weight_v = 1,
        weight_t = 1,
        weight_avt = 1.
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
                loss = self.training_step( model=model, batch=batch, 
                                                              batch_idx=batch_idx, 
                                                              weight_a= weight_a,
                                                              weight_v= weight_v,
                                                              weight_t= weight_t,
                                                              weight_avt= weight_avt
                                                              )
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
    
    def training_step(self, model, batch, batch_idx, weight_a, weight_v, weight_t, weight_avt):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        if self.modality == 3:
            out_a, out_v, out_t, out = model(batch)
            label = batch['label']
            label = label.to(model.device)
            model.train()
            loss_a = criterion(out_a, label)
            loss_v = criterion(out_v, label)
            loss_t = criterion(out_t, label)
            loss_avt = criterion(out, label)

            if self.modulation_starts <= self.current_epoch < self.modulation_ends:
                loss = weight_a * loss_a + weight_v * loss_v + weight_avt * loss_avt +weight_t * loss_t    
            else:
                loss = loss_avt
            
        elif self.modality == 2:
            out_a, out_v, out = model(batch)
            label = batch['label']
            label = label.to(model.device)
            model.train()
            loss_a = criterion(out_a, label)
            loss_v = criterion(out_v, label)
            loss_avt = criterion(out, label)

            if self.modulation_starts <= self.current_epoch < self.modulation_ends:
                loss = weight_a * loss_a + weight_v * loss_v + weight_avt * loss_avt  
            else:
                loss = loss_avt
        loss.backward()
        return loss
    def super_training_step(self, model, batch, batch_idx, types = 0):
        out = model(batch, batch_idx,types = types)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, batch['label'].to(model.device))
        
        return loss
                
    def super_epoch_origin(self, model, temp_model, limit_batches, train_loader, test_loader, optimizer, logger):
        pre_a_loss_train = 0.0
        pre_v_loss_train = 0.0
        pre_t_loss_train = 0.0
        pre_av_loss_train = 0.0
        pre_avt_loss_train = 0.0

        now_a_loss_train = 0.0
        now_v_loss_train = 0.0
        now_t_loss_train =0.0
        now_av_loss_train = 0.0
        now_avt_loss_train = 0.0

        pre_a_loss_test = 0.0
        pre_v_loss_test = 0.0
        pre_t_loss_test = 0.0
        pre_av_loss_test = 0.0
        pre_avt_loss_test = 0.0

        now_a_loss_test = 0.0
        now_v_loss_test = 0.0
        now_t_loss_test = 0.0
        now_av_loss_test = 0.0
        now_avt_loss_test = 0.0
        _loss_avt = 0.0
        _loss_av = 0.0
        _loss_a = 0.0
        _loss_v = 0.0
        _loss_t = 0.0
        _loss_av_test = 0.0
        _loss_a_test = 0.0
        _loss_v_test = 0.0
        _loss_t_test = 0.0
        _loss_avt_test = 0.0
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        print('super start')
        # train_dataloader = self.progbar_wrapper(
        #     train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        # )
        train_dataloader = train_loader
        test_dataloader = test_loader
        # test_dataloader = self.progbar_wrapper(
        #     test_loader, total=min(len(test_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        # )
        #audio flow
        temp_model.load_state_dict(model.state_dict(),strict=True)
        for epoch in range(self.super_epoch):
            _loss_a = 0.0
            _loss_a_test = 0.0
            temp_model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                # image = image.to(device)
                loss_a = self.super_training_step(temp_model,batch, batch_idx, types = 1)
                _loss_a += loss_a.item()
                optimizer.step()


            _loss_a /= len(train_dataloader)

            if epoch == 0 or epoch == self.super_epoch - 1:
                with torch.no_grad():
                    temp_model.eval()
                    for batch_idx, batch in enumerate(test_dataloader):
                        loss_a = self.super_training_step(model, batch, batch_idx, types = 1)
                        _loss_a_test += loss_a.item()
                    _loss_a_test /= len(test_dataloader)

                    if epoch == 0:
                        pre_a_loss_train = _loss_a
                        pre_a_loss_test = _loss_a_test
                    else:
                        now_a_loss_train = _loss_a
                        now_a_loss_test = _loss_a_test

        # visual
        temp_model.load_state_dict(model.state_dict(),strict=True)
        for epoch in range(self.super_epoch):
            _loss_v = 0.0
            _loss_v_test = 0.0
            temp_model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                # image = image.to(device)
                loss_v = self.super_training_step(temp_model,batch, batch_idx, types = 2)
                _loss_v += loss_v.item()
                optimizer.step()

            _loss_v /= len(train_dataloader)

            if epoch == 0 or epoch == self.super_epoch - 1:
                with torch.no_grad():
                    temp_model.eval()
                    for batch_idx, batch in enumerate(test_dataloader):

                        loss_v = self.super_training_step(model, batch, batch_idx, types = 2)

                        _loss_v_test += loss_v.item()

                    _loss_v_test /= len(test_dataloader)

                    if epoch == 0:
                        pre_v_loss_train = _loss_v
                        pre_v_loss_test = _loss_v_test
                    else:
                        now_v_loss_train = _loss_v
                        now_v_loss_test = _loss_v_test

        # text
        if self.modality == 3:
            temp_model.load_state_dict(model.state_dict(),strict=True)
            for epoch in range(self.super_epoch):
                _loss_t = 0.0
                _loss_t_test = 0.0
                temp_model.train()
                for batch_idx, batch in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    # image = image.to(device)
                    loss_t = self.super_training_step(temp_model,batch, batch_idx, types = 3)
                    _loss_t += loss_t.item()
                    optimizer.step()

                _loss_t /= len(train_dataloader)

                if epoch == 0 or epoch == self.super_epoch - 1:
                    with torch.no_grad():
                        temp_model.eval()
                        for batch_idx, batch in enumerate(test_dataloader):

                            loss_t = self.super_training_step(model, batch, batch_idx, types = 3)
                            _loss_t_test += loss_t.item()

                        _loss_t_test /= len(test_dataloader)

                        if epoch == 0:
                            pre_t_loss_train = _loss_t
                            pre_t_loss_test = _loss_t_test
                        else:
                            now_t_loss_train = _loss_t
                            now_t_loss_test = _loss_t_test

        # all
        temp_model.load_state_dict(model.state_dict(),strict=True)
        for epoch in range(self.super_epoch):
            temp_model.train()
            _loss_avt = 0.0
            _loss_avt_test = 0.0
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                label = batch['label'].to(model.device)
                # image = image.to(device)
                if self.modality == 3:
                    _, _, _, out_avt = temp_model(batch, batch_idx)
                else:
                    _, _, out_avt = temp_model(batch, batch_idx)
                loss_avt = criterion(out_avt, label)
                loss_avt.backward()
                _loss_avt += loss_avt.item()
                optimizer.step()

            _loss_avt /= len(train_dataloader)

            if epoch == 0 or epoch == self.super_epoch - 1:
                with torch.no_grad():
                    temp_model.eval()
                    for batch_idx,batch in enumerate(test_dataloader):
                        label = batch['label'].to(model.device)
                        if self.modality == 3:
                            _, _, _, out_avt = temp_model(batch, batch_idx)
                        else:
                            _, _, out_avt = temp_model(batch, batch_idx)
                        loss_avt = criterion(out_avt, label)
                        _loss_avt_test += loss_avt.item()

                    _loss_avt_test /= len(test_dataloader)

                    if epoch == 0:
                        pre_avt_loss_train = _loss_avt
                        pre_avt_loss_test = _loss_avt_test
                    else:
                        now_avt_loss_train = _loss_avt
                        now_avt_loss_test = _loss_avt_test
            
        g_a = pre_a_loss_test - now_a_loss_test
        o_a_pre = pre_a_loss_test - pre_a_loss_train
        o_a_now = now_a_loss_test - now_a_loss_train
        o_a = o_a_now - o_a_pre
        weight_a = abs(g_a / (o_a * o_a))

        g_v = pre_v_loss_test - now_v_loss_test
        o_v_pre = pre_v_loss_test - pre_v_loss_train
        o_v_now = now_v_loss_test - now_v_loss_train
        o_v = o_v_now - o_v_pre
        weight_v = abs(g_v / (o_v * o_v))

        if self.modality == 3:
            g_t = pre_t_loss_test - now_t_loss_test
            o_t_pre = pre_t_loss_test - pre_t_loss_train
            o_t_now = now_t_loss_test - now_t_loss_train
            o_t = o_t_now - o_t_pre
            weight_t = abs(g_t / (o_t * o_t))
        else:
            weight_t = 0.0

        g_avt = pre_avt_loss_test - now_avt_loss_test
        o_avt_pre = pre_avt_loss_test - pre_avt_loss_train
        o_avt_now = now_avt_loss_test - now_avt_loss_train
        o_avt = o_avt_now - o_avt_pre
        weight_avt = abs(g_avt / (o_avt * o_avt))

        sums = weight_a + weight_v + weight_avt + weight_t

        weight_a /= sums
        weight_v /= sums
        weight_avt /= sums
        weight_t /= sums
        logger.info("super epoch in epoch {:0}".format(self.current_epoch))
        logger.info("pre_a_loss_train:{:0}, pre_v_loss_train:{:1}, pre_avt_loss_train:{:2}".format(pre_a_loss_train, pre_v_loss_train, pre_avt_loss_train))
        logger.info("now_a_loss_train:{:0}, now_v_loss_train:{:1}, now_avt_loss_train:{:2}".format(now_a_loss_train, now_v_loss_train, now_avt_loss_train))
        return weight_a, weight_v, weight_t, weight_avt