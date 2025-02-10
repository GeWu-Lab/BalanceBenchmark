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
from lightning_utilities import apply_to_collection
import lightning as L
import torch



class GSPlugin():
    def __init__(self, device, gs_flag = True ):

        super().__init__()

        # dtype = torch.cuda.FloatTensor  # run on GPU
        if device != ' ':
            device = 'cuda:0'
        else:
            device = 'cpu'
        with torch.no_grad():
            # self.Pl = torch.eye(1024).to(device)
            self.Pl = torch.eye(512).to(device)
        self.exp_count = 0

    # @torch.no_grad()
    def before_update(self, model, before_batch_input, batch_index, len_dataloader, train_exp_counter):
        lamda = batch_index / len_dataloader + 1
        alpha = 1.0 * 0.1 ** lamda
        
        # x_mean = torch.mean(strategy.mb_x, 0, True)
        if train_exp_counter != 0:
            for n, w in model.named_parameters():
                if n == "weight":
                    
                    r = torch.mean(before_batch_input, 0, True)
                    k = torch.mm(self.Pl, torch.t(r))
                    self.Pl = torch.sub(self.Pl, torch.mm(k, torch.t(k)) / (alpha + torch.mm(k, r)))

                    pnorm2 = torch.norm(self.Pl.data, p='fro')

                    self.Pl.data = self.Pl.data / pnorm2
                    w.grad.add_(torch.mm(w.grad.data, torch.t(self.Pl.data)))
                    # grad_update = torch.mm(w.grad.data, torch.t(self.Pl.data))
                    # if w.grad is None:
                        # w.grad = grad_update
                    # else:
                        # w.grad = w.grad + grad_update
                   
                    
                    
class MLATrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {},args = {}):
        super(MLATrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.device = args.model['device']
        self.gs_plugin = GSPlugin(self.device) 
        self.criterion = nn.CrossEntropyLoss()
        
        
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
        modality_list = model.modalitys
        all_modalitys = list(model.modalitys)
        all_modalitys.append('output')
        self.PrecisionCalculator = self.PrecisionCalculatorType(model.n_classes, all_modalitys)
        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )
        len_dataloader = len(train_loader)
        self.fabric.call("on_train_epoch_start")

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                self.training_step(model=model, batch=batch, batch_idx=batch_idx,len_dataloader=len_dataloader,optimizer=optimizer)

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx,len_dataloader=len_dataloader)
            # self.PrecisionCalculator.update(y_true = batch['label'].cpu(), y_pred = model.pridiction)
            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            # if should_optim_step:
            #     self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            self._format_iterable(iterable, self._current_train_return, "train")
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)
            
        # self._current_metrics = self.PrecisionCalculator.compute_metrics()
        self.fabric.call("on_train_epoch_end")
    
    def training_step(self, model, batch, batch_idx,len_dataloader,optimizer):

        # TODO: make it simpler and easier to extend
        modality_list = model.modalitys
        key = list(modality_list)
        # out_v,out_a,out = Uni_res['visual'], Uni_res['audio'], Uni_res['output']
        label = batch['label']
        # label = label.to(model.device)
        loss = 0
        loss_modality = {}
        # feature =model(batch)
        if  self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            # feature = {}
            # for modality in modality_list:
            #     feature[modality] = model.encoder_res[modality].clone().contiguous()
            for modality in modality_list:
                feature = model.feature_extract(batch, modality = modality)
                out = model.fusion_module.fc_out(feature)
                loss = self.criterion(out,label)
                # print("12345")
                try:
                    loss.backward()
                except RuntimeError as e:
                    # 打印计算图信息
                    print("Computation graph:")
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(f"{name} grad shape: {param.grad.shape}")
                    raise e
                # print("678910")
                encoder_out = feature.detach()
                self.gs_plugin.before_update(model.fusion_module.fc_out, encoder_out, 
                                    batch_idx, len_dataloader, self.gs_plugin.exp_count)
                
                self.fabric.call("on_before_optimizer_step", optimizer, 0)   
                optimizer.step()
                self.fabric.call("on_before_zero_grad", optimizer)
                optimizer.zero_grad()  
                loss_modality[modality] = loss.item()
                self.gs_plugin.exp_count += 1
                
            for n, p in model.named_parameters():
                if p.grad != None:
                    del p.grad
          
            loss = self.alpha*loss_modality[key[0]]+(1-self.alpha)*loss_modality[key[1]]
            
                
        else:
            loss = self.criterion(model.Uni_res['output'], label)
            loss.backward()
        

        # # avoid gradients in stored/accumulated values -> prevents potential OOM
        # self._current_train_return = apply_to_collection(model.encoder_res, dtype=torch.Tensor, function=lambda x: x.detach())
        return loss
    