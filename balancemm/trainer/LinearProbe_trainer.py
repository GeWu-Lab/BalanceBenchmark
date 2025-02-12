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
from lightning_utilities import apply_to_collection

from ..models.avclassify_model import BaseClassifierModel
from ..evaluation.precisions import BatchMetricsCalculator
from ..evaluation.complex import get_flops
from ..models import create_model
import copy
from ..utils.train_utils import get_checkpoint_files, get_newest_path
import os.path as osp
from lightning.fabric.loggers import TensorBoardLogger
from ..evaluation.modalitys import Calculate_Shapley
from logging import Logger
class NewLinearHead(nn.Module):
    '''New Linear-Classifier-Head'''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.unimodal_result = {}
        self.prediction = {}
    def forward(self, encoder_result):
        output = torch.cat(list(encoder_result.values()), dim=1)
        output = self.fc_out(output)
        return output
        
    def Unimodalitiy_Calaulate(self, encoder_result, modality_size):
        softmax = softmax =nn.Softmax(dim= 1)
        now_size = 0
        all_nums = len(encoder_result.keys())-1
        
        for modality in encoder_result.keys():
            if modality == 'output':
                self.unimodal_result[modality] = encoder_result[modality]
                continue
                
            weight_size = self.fc_out.weight.size(1)
            self.unimodal_result[modality] = (torch.mm(encoder_result[modality],
                                        torch.transpose(self.fc_out.weight[:,
                                                                        now_size:
                                                                        now_size + modality_size[modality]], 0, 1))
                            + self.fc_out.bias / all_nums)
            now_size += modality_size[modality]
        for modality in encoder_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        
        return self.unimodal_result
class LinearProbeTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}, args = {}):
        super(LinearProbeTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.trainer_probed = method_dict['trainer_probed']
        
        # finding the newest trained model
        temp_args = copy.deepcopy(args)
        out_dir = '_'.join(temp_args.out_dir.split('/')[:-1])
        out_dir = temp_args.out_dir.replace('LinearProbeTrainer', self.trainer_probed)
        out_dir = '/'.join(out_dir.split('/')[:-3])
        out_dir = os.path.join(out_dir,"train_and_test")

        self.checkpoint_path = get_newest_path(out_dir)
        

    
    def fit(
        self,
        model,
        new_head,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: torch.optim.lr_scheduler,
        logger: Logger,
        tb_logger: TensorBoardLogger,
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
       
        self.precision_calculator = self.PrecisionCalculatorType(model.n_classes, model.modalitys)
        # setup dataloaders
        if self.should_train:
            train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)

        # Loading the newest trained model
        model.load_state_dict(torch.load(get_checkpoint_files(self.checkpoint_path)[0])['model'])
        # assemble state (current epoch and global step will be added in save)
        state = {"model": model,"new_head": new_head, "optim": optimizer, "scheduler": scheduler_cfg}

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True
        modality_list = model.modalitys
        tb = {}
        if tb_logger:
            for modality in modality_list:
                tb[modality] = TensorBoardLogger(root_dir=tb_logger.root_dir, name=f'tensorboard',default_hp_metric=False,version=0,sub_dir = f'{modality}')
        for param in model.parameters():
            param.requires_grad = False
        
        while not self.should_stop:
            if self.should_train:
                model.eval()
                new_head.train()
                self.train_loop(
                    model,new_head, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg
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
                            tag = "train_acc"
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
                new_head.eval()
                valid_loss, Metrics_res =self.val_loop(model,new_head, val_loader, limit_batches=self.limit_val_batches)
                info = f'valid_loss: {valid_loss}'
                output_info = ''
                if tb_logger:
                    tb_logger.log_metrics({
                        'valid_loss': valid_loss,
                    }, step=self.current_epoch)
                
               
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
            
            if scheduler_cfg is not None:
                scheduler_cfg.step()

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True
            
            if self.should_save and self.should_train:
                self.save(state)
                self.should_save = False

        # reset for next fit call
        self.should_stop = False
        
    def train_loop(
        self,
        model: L.LightningModule,
        new_head: NewLinearHead,
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
                optimizer.step(partial(self.training_step, model=model,new_head = new_head, batch=batch, batch_idx=batch_idx))
                self.fabric.call("on_before_zero_grad", optimizer)

                optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model,new_head=new_head, batch=batch, batch_idx=batch_idx)

            self.precision_calculator.update(y_true = batch['label'].cpu(), y_pred = new_head.prediction)
            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
           
            # add output values to progress bar
            
            self._format_iterable(iterable, self._current_train_return, "train")
            
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)
        self._current_metrics = self.precision_calculator.compute_metrics()
        self.fabric.call("on_train_epoch_end")
    def training_step(self, model: BaseClassifierModel, new_head: NewLinearHead,batch, batch_idx):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        
        label = batch['label']
        label = label.to(model.device)
       
        with torch.no_grad():
            model(batch= batch)
        encoder_features = {k: v for k, v in model.encoder_result.items() if k != 'output'}
        out = new_head(encoder_features)
        encoder_features['output'] = out   
        new_head.unimodal_result = new_head.Unimodalitiy_Calaulate(encoder_features,model.modality_size)
        loss = criterion(out, label)

        loss.backward()
        return loss
    
    def val_loop(
        self,
        model: BaseClassifierModel,
        new_head: NewLinearHead,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
        limit_modalitys: list = ['ALL']
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

            out = self.validation_step(model,new_head,batch, batch_idx,limit_modalitys)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")
            
            MetricsCalculator.update(y_true = batch['label'].cpu(), y_pred = new_head.prediction)
            valid_loss += out
         
        valid_loss /= MetricsCalculator.total_samples
        Metrics_res = MetricsCalculator.compute_metrics()
        self._current_metrics = Metrics_res
        self.best_acc={}
        self.best_acc['output'] = 0
        if Metrics_res['acc']['output'] > self.best_acc['output']:
            self.should_save = True
            self.best_acc['output'] = Metrics_res['acc']['output']
            for modality in model.modalitys:
                self.best_acc[modality] = Metrics_res['acc'][modality]

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)
        return valid_loss, Metrics_res
    
    def validation_step(self, model, new_head,batch, batch_idx,limit_modalitys):
    
        padding = []
        for modality in model.modalitys:
            if modality not in limit_modalitys:
                padding.append(modality)
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            model(batch,padding=padding)
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        label = label.to(model.device)
        key = list(model.modalitys)
        modality_list = model.modalitys
        encoder_features = {}
        for modality in modality_list:
            encoder_features[modality] = model.encoder_result[modality]
        encoder_features['output'] = new_head(encoder_features)
        new_head.unimodal_result = new_head.Unimodalitiy_Calaulate(encoder_features,model.modality_size)
        loss = criterion(encoder_features['output'], label)
        return loss