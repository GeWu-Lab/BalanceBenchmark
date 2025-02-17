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
from ..evaluation.modalitys import Calculate_Shapley
from lightning.fabric.loggers import TensorBoardLogger
class GBlendingTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(GBlendingTrainer,self).__init__(fabric,**para_dict)
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        
        self.super_epoch = method_dict['super_epoch']
        self.modality = method_dict['modality']
        if self.method == 'offline':
            self.super_epoch = self.max_epochs 

    def fit(
        self,
        model,
        temp_model,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: torch.optim.lr_scheduler,
        temp_optimizer: torch.optim.Optimizer,
        temp_optimizer_origin,
        logger,
        tb_logger,
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

        weights = {}
        tb = {}
        if tb_logger:
            for modality in model.modalitys:
                tb[modality] = TensorBoardLogger(root_dir=tb_logger.root_dir, name=f'tensorboard',default_hp_metric=False,version=0,sub_dir = f'{modality}')
        Shapley = {}
        while not self.should_stop:
            if tb_logger:
                tb_logger.log_hyperparams({"epochs": self.current_epoch})
            if self.current_epoch % self.super_epoch == 0:
                model.train()

                weights = self.super_epoch_origin(model, temp_model, self.limit_train_batches, train_loader, val_loader, temp_optimizer,logger,temp_optimizer_origin)
                logger.info(weights)
                if tb_logger:
                    for modality in weights.keys():
                        tb_logger.log_metrics({
                            f"super weights_{modality}": weights[modality]
                        }, step=self.current_epoch)
            model.train()
            self.train_loop(
                model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg,
                weights = weights
            )
            logger.info("epoch: {:0}  ".format(self.current_epoch))
            if tb_logger:
                tb_logger.log_hyperparams({"epochs": self.current_epoch})
            output_info = ''
            info = ''
            ##parse the Metrics
            Metrics_res = self._current_metrics
            for metircs in sorted(Metrics_res.keys()):
                if metircs == 'acc':
                    valid_acc = Metrics_res[metircs]
                    for modality in sorted(valid_acc.keys()):
                        tag = 'train_acc'
                        if modality == 'output':
                            output_info += f"train_acc: {valid_acc[modality]}"
                            tb_logger.log_metrics({
                                tag: valid_acc[modality]
                            }, step=self.current_epoch)
                        else:
                            info += f", acc_{modality}: {valid_acc[modality]}"
                            tb[modality].log_metrics({
                                tag: valid_acc[modality]
                            }, step=self.current_epoch)
                        
                if metircs == 'f1':
                    valid_f1 = Metrics_res[metircs]
                    for modality in sorted(valid_f1.keys()):
                        tag = "train_f1"
                        if modality == 'output':
                            output_info += f", train_f1: {valid_f1[modality]}"
                            tb_logger.log_metrics({
                                tag: valid_f1[modality]
                            }, step=self.current_epoch)
                        else:
                            info += f", f1_{modality}: {valid_f1[modality]}"
                        
                            tb[modality].log_metrics({
                                tag: valid_f1[modality]
                            }, step=self.current_epoch)
                info = output_info+ ', ' + info
                    
                logger.info(info)
                self.precision_calculator.ClearAll()
            if self.should_validate:
                model.eval()
                valid_loss, Metrics_res =self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)
                info = f'valid_loss: {valid_loss}'
                output_info = ''
                if tb_logger:
                    tb_logger.log_metrics({
                        'valid_loss': valid_loss,
                    }, step=self.current_epoch)
                # ##parse the Metrics
                # Shapley = Calculate_Shapley(self, model,val_loader,logger)
                # for modality in model.modalitys:
                #     tag = "Shapley_value"
                #     tb[modality].log_metrics({
                #                     tag: Shapley[modality]
                #                 }, step=self.current_epoch)
                for metircs in sorted(Metrics_res.keys()):
                    if metircs == 'acc':
                        valid_acc = Metrics_res[metircs]
                        for modality in sorted(valid_acc.keys()):
                            tag = "valid_acc"
                            if modality == 'output':
                                output_info += f"valid_acc: {valid_acc[modality]}"
                                tb_logger.log_metrics({
                                    tag: valid_acc[modality]
                                }, step=self.current_epoch)
                            else:
                                info += f", acc_{modality}: {valid_acc[modality]}"
                            
                                tb[modality].log_metrics({
                                    tag: valid_acc[modality]
                                }, step=self.current_epoch)
                                
                    if metircs == 'f1':
                        valid_f1 = Metrics_res[metircs]
                        for modality in sorted(valid_f1.keys()):
                            tag = "valid_f1"
                            if modality == 'output':
                                output_info += f", valid_f1: {valid_f1[modality]}"
                                tb_logger.log_metrics({
                                    tag: valid_f1[modality]
                                }, step=self.current_epoch)
                            else:
                                info += f", f1_{modality}: {valid_f1[modality]}"
                           
                                tb[modality].log_metrics({
                                    tag: valid_f1[modality]
                                }, step=self.current_epoch)
                info = output_info+ ', ' + info
                    
                logger.info(info)
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
        weights : dict = None
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
                loss = self.training_step( model=model, batch=batch, 
                                                              batch_idx=batch_idx, 
                                                              weights = weights
                                                              )
                optimizer.step()
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
    
    def training_step(self, model: BaseClassifierModel, batch, batch_idx, weights):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        model(batch)
        label = batch['label']
        label = label.to(model.device)
        loss = 0
        for modality in model.unimodal_result.keys():
            loss += weights[modality] * criterion(model.unimodal_result[modality], label)
        loss.backward()
        return loss
    
                
    def super_epoch_origin(self, model, temp_model, limit_batches, train_loader, test_loader, optimizer, logger,optimizer_origin):
        temp_model.load_state_dict(model.state_dict(),strict=True)
        criterion = nn.CrossEntropyLoss()
        weights = {}
        modalitys = list(temp_model.modalitys)
        modalitys.append('output')
        for modality in modalitys:
            temp_model.load_state_dict(model.state_dict(),strict=True)
            optimizer.load_state_dict(optimizer_origin)
            padding = []
            pre_train_loss = 0
            now_train_loss = 0
            pre_validation_loss = 0
            now_validation_loss = 0
            print(modality)
            for other_modality in temp_model.modalitys:
                if other_modality != modality and modality !='output':
                    padding.append(other_modality)
            for epoch in range(self.super_epoch):
                temp_model.train()
                _loss = 0.0
                train_dataloader = self.progbar_wrapper(
                train_loader, total=min(len(train_loader), limit_batches), desc=f"Super_Epoch {epoch}"
                )
                if epoch == 0 or epoch == self.super_epoch - 1 :
                    test_dataloader = self.progbar_wrapper(
                        test_loader, total=min(len(test_loader), limit_batches), desc=f"Validation in Super_epoch {epoch}"
                    )
                for batch_idx, batch in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    temp_model(batch, padding = padding)
                    out = temp_model.unimodal_result['output']
                    loss = criterion(out, batch['label'].to(temp_model.device))
                    loss.backward() #
                    optimizer.step()
                    _loss += loss.item()
                if epoch == 0 or epoch == self.super_epoch - 1 :
                    temp_model.eval()
                    with torch.no_grad():
                        _loss_v = 0
                        for batch_idx, batch in enumerate(test_dataloader):
                            temp_model(batch, padding = padding)
                            out = temp_model.unimodal_result['output']
                            loss = criterion(out, batch['label'].to(temp_model.device))
                            _loss_v += loss.item()
                        if epoch == 0:
                            pre_train_loss = _loss
                            pre_validation_loss = _loss_v
                            print(f"valid_loss is {_loss_v} on super epoch {epoch}")
                            logger.info(f"valid_loss is {_loss_v} on super epoch {epoch}")
                        else:
                            now_train_loss = _loss
                            now_validation_loss = _loss_v
                            print(f"valid_loss is {_loss_v} on super epoch {epoch}")
                            logger.info(f"valid_loss is {_loss_v} on super epoch {epoch}")
            g = now_validation_loss - pre_validation_loss
            o_pre = pre_validation_loss - pre_train_loss
            o_now = now_validation_loss - now_train_loss
            o = o_now - o_pre

            weights[modality] = abs((g )/(o**2))

        sums = sum(weights.values() ) 
        info = ''
        logger.info(f'super_epoch begin in {self.current_epoch}')
        for modality in weights.keys():
            weights[modality]/=sums
            info += f'{modality} weight is {weights[modality]} || '
        logger.info(info)
       
        return weights