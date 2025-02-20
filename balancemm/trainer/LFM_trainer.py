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
from ..models.avclassify_model import BaseClassifierModel

from ..evaluation.complex import FLOPsMonitor
from ..evaluation.modalitys import Calculate_Shapley
from logging import Logger
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger

def manual_combinations(n):
    result = []
    for i in range(0, n-1):
        for j in range(i + 1, n):
            result.append((i, j))
    return result


def Alignment(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * p + 0.5 * q
    kl_p_m = F.kl_div(p.log(), m, reduction='batchmean')
    kl_q_m = F.kl_div(q.log(), m, reduction='batchmean')
    js_score = 0.5 * (kl_p_m + kl_q_m)
    return js_score

def getAlpha_Learnable(epoch, valbatch, model, alpha, lr_alpha=1e-4):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    
    half_size = valbatch['label'].size(0)//4
    new_batch = {key: value[:half_size] for key,value in valbatch.items()}
    
    model(new_batch)
    modality_num = len(model.modalitys)
    combinations = manual_combinations(modality_num)
    loss_alignment = 0
    modalitys = list(model.modalitys)
    for combination in combinations:
        loss_alignment += Alignment(model.unimodal_result[modalitys[combination[0]]],model.unimodal_result[modalitys[combination[1]]])
    out = model.unimodal_result['output']        
    loss_cls = criterion(out, new_batch['label']).mean()
    L_total = (alpha[0]* loss_cls + alpha[1]* loss_alignment/len(combinations))
    theta_grads = torch.autograd.grad(L_total, model.parameters(), create_graph=True)

    cls_grads = torch.autograd.grad(loss_cls, model.parameters(), retain_graph=True, create_graph=True)


    hessian_vector_prod = torch.autograd.grad(
        outputs=theta_grads,
        inputs=model.parameters(),
        grad_outputs=cls_grads,
        retain_graph=True
    )

    with torch.no_grad():
        alpha_grad = hessian_vector_prod[0].mean().item()
        print(alpha_grad)
        
        if epoch % 20 == 0 and epoch != 0:
            lr_alpha = lr_alpha / 2 
        alpha[1] -= lr_alpha * alpha_grad
        alpha[1] = max(0.05, min(0.95, alpha[1])) 
        alpha[0] = 1.0 - alpha[1] 


    return alpha, lr_alpha
class LFMTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(LFMTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.modality = method_dict['modality']
        self.lr_alpha = method_dict['lr_alpha']

    def fit(
        self,
        model,
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
        # self.fabric.launch()
        # setup calculator
        self.precision_calculator = self.PrecisionCalculatorType(model.n_classes, model.modalitys)
        # setup dataloaders
        if self.should_train:
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
        modality_list = model.modalitys
        tb = {}
        if tb_logger:
            for modality in modality_list:
                tb[modality] = TensorBoardLogger(root_dir=tb_logger.root_dir, name=f'tensorboard',default_hp_metric=False,version=0,sub_dir = f'{modality}')
        Shapley = {}
        val_batch = next(iter(val_loader))
        alpha = {1:0.05,0:0.95}
        while not self.should_stop:
            if self.should_train:
                model.train()
                self.alpha, self.lr_alpha = getAlpha_Learnable(self.current_epoch, valbatch=val_batch, model= model, alpha = alpha, lr_alpha = self.lr_alpha)
                self.train_loop(
                    model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg
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
                valid_loss, Metrics_res =self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)
                info = f'valid_loss: {valid_loss}'
                output_info = ''
                if tb_logger:
                    tb_logger.log_metrics({
                        'valid_loss': valid_loss,
                    }, step=self.current_epoch)
                # with torch.no_grad():
                #     Shapley = Calculate_Shapley(self, model,val_loader,logger)
                # for modality in modality_list:
                #     tag = "Shapley_value"
                #     tb[modality].log_metrics({
                #                     tag: Shapley[modality]
                #                 }, step=self.current_epoch)
                ##parse the Metrics
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
    def training_step(self, model: BaseClassifierModel, batch, batch_idx):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        label = batch['label']
        label = label.to(model.device)
        model(batch)
        modality_num = len(model.modalitys)
        combinations = manual_combinations(modality_num)
        loss_alignment = 0
        modalitys = list(model.modalitys)
        for combination in combinations:
            loss_alignment += Alignment(model.unimodal_result[modalitys[combination[0]]],model.unimodal_result[modalitys[combination[1]]])
        out = model.unimodal_result['output']        
        loss = criterion(out, label)
        loss = 2 * (self.alpha[0]* loss + self.alpha[1]* loss_alignment/len(combinations))
        loss.backward()
        return loss