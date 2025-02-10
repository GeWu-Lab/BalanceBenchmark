import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from tqdm import tqdm
from ..evaluation.precisions import BatchMetricsCalculator
from ..models.avclassify_model import BaseClassifierModel
from ..evaluation.complex import FLOPsMonitor
from ..evaluation.modalitys import Calculate_Shapley
from logging import Logger
from .base_trainer import BaseTrainer
    
class ReconBoostTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(ReconBoostTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.T_epochs = method_dict['T_epochs']
        self.weight1 = method_dict['weight1']
        self.weight2 = method_dict['weight2']
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
        self.PrecisionCalculator = self.PrecisionCalculatorType(model.n_classes, model.modalitys)
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
        key = list(model.modalitys)
        
        while not self.should_stop:
            if self.should_train:
                model.train()
                    
                for modality in model.modalitys:
                    
                    for i in range(len(key)):
                        if key[i] == modality:
                            if i == 0 :
                                pre_modality = key[len(key)-1]
                            else:
                                pre_modality = key[i-1]
                    self.train_loop(
                        model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg, modality=modality, pre_modality=pre_modality
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
                self.PrecisionCalculator.ClearAll()
            if self.should_validate:
                model.eval()
                
                valid_loss, Metrics_res =self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)
                info = f'valid_loss: {valid_loss}'
                output_info = ''
                if tb_logger:
                    tb_logger.log_metrics({
                        'valid_loss': valid_loss,
                    }, step=self.current_epoch)
                # if self.current_epoch == self.max_epochs-1:
                #     rng_state = torch.get_rng_state()
                #     cuda_rng_state = torch.cuda.get_rng_state()    
                #     Calculate_Shapley(self, model,val_loader,logger,is_print=True)
                #     torch.set_rng_state(rng_state)
                #     torch.cuda.set_rng_state(cuda_rng_state)  
                # rng_state = torch.get_rng_state()
                # cuda_rng_state = torch.cuda.get_rng_state()    
                # Shapley = Calculate_Shapley(self, model,val_loader,logger)
                # torch.set_rng_state(rng_state)
                # torch.cuda.set_rng_state(cuda_rng_state)   
                # for modality in modality_list:
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
                self.PrecisionCalculator.ClearAll()
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
        modality = None,
        pre_modality = None
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
        
        # phase 1
        for epoch in range(self.T_epochs):
            self.fabric.call("on_train_epoch_start")
            iterable = self.progbar_wrapper(
                train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch} _{modality}_train"
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
                    

                    # optimizer step runs train step internally through closure
                    self.training_step1(model=model, batch=batch, batch_idx=batch_idx,modality=modality,pre_modality = pre_modality)
                    self.fabric.call("on_before_optimizer_step", optimizer, 0)
                    optimizer.step()
                    self.fabric.call("on_before_zero_grad", optimizer)
                    optimizer.zero_grad()
                        
                        
                else:
                    # gradient accumulation -> no optimizer step
                    self.training_step2(model=model, batch=batch, batch_idx=batch_idx)
                # self.PrecisionCalculator.update(y_true = batch['label'].cpu(), y_pred = model.pridiction)
                self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

                # this guard ensures, we only step the scheduler once per global step
                # if should_optim_step:
                #     self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

                # add output values to progress bar
                
                self._format_iterable(iterable, self._current_train_return, "train")
                
                # only increase global step if optimizer stepped
                self.global_step += int(should_optim_step)

        # phase 2
        for epoch in range(self.T_epochs):
            all_modalitys = list(model.modalitys)
            all_modalitys.append('output')
            self.PrecisionCalculator = self.PrecisionCalculatorType(model.n_classes, all_modalitys)
            iterable = self.progbar_wrapper(
                train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch} _ train"
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
                    

                    # optimizer step runs train step internally through closure
                    self.training_step2(model=model, batch=batch, batch_idx=batch_idx)
                    self.fabric.call("on_before_optimizer_step", optimizer, 0)
                    optimizer.step()
                    self.fabric.call("on_before_zero_grad", optimizer)
                    optimizer.zero_grad()
                        
                        
                else:
                    # gradient accumulation -> no optimizer step
                    self.training_step2(model=model, batch=batch, batch_idx=batch_idx)
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
    
    def training_step1(self, model, batch, batch_idx, modality,pre_modality,dependent_modality : str = 'none'):

        # TODO: make it simpler and easier to extend
        mse_criterion = nn.MSELoss()
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)
      
        label = batch['label']
 
        # device = model.device
        
        # modality_data = batch[modality].to(model.device)
        # pre_modality_data = batch[pre_modality].to(model.device)
        modality_data = batch[modality]
        pre_modality_data = batch[pre_modality]
       
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            model.forward(batch, mask_model=modality) ## mask_model = model_name global_ft: true or false
            out_join = model.Uni_res['output']
            # out_obj = model.modality_model(modality_data,modality = modality,)
            out_obj = model.modality_model(batch,modality = modality,)
            target = torch.zeros(label.size(0),model.n_classes).to(model.device).scatter_(1,label.view(-1,1),1)
            boosting_loss = - self.weight1 *  (target * log_softmax(out_obj)).mean(-1) \
                        + self.weight2 * (target * softmax(out_join.detach()) * log_softmax(out_obj)).mean(-1)
            
            if self.current_epoch == 0:
                loss = boosting_loss
            else:
                pre_out_obj = model.modality_model(batch,modality = pre_modality)
                ga_loss = mse_criterion(softmax(out_obj.detach()), softmax(pre_out_obj.detach())) ## ga loss
                loss = boosting_loss + self.alpha * ga_loss
            loss.mean().backward()
                
        else:
        
            total_loss = loss['out']
            total_loss.backward()

        return loss
    
    def training_step2(self, model, batch, batch_idx , dependent_modality : str = 'none'):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)
        # out_a, out_v = model.AVCalculate(a, v, out)
        label = batch['label']
        # device = model.device
        # print(a.shape, v.shape, model.head.weight.shape)
        ## our modality-wise normalization on weight and feature
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            model.forward_grad(batch) ## mask_model = model_name global_ft: true or false
            out = model.Uni_res['output']
            loss = criterion(out,label)
            loss.backward()
                
        else:
            # model(batch)
            # for modality in modality_list:
            #     m[modality] = model.encoder_res[modality]
            # m['out'] = model.encoder_res['output']
            # out_a, out_v = model.AVCalculate(a, v, out)
        
            total_loss = loss['out']
            total_loss.backward()

        return loss