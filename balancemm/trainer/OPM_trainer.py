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
import numpy as np


class Modality_drop():
    def __init__(self, dim_list, p_exe=0.7, device='cuda'):
        self.dim_list = dim_list
        self.p_exe = p_exe
        self.device = device
        
    def execute_drop(self, feat_list, q, model):
        modality_list = list(model.modalitys)
        B = feat_list[modality_list[0]].shape[0]  # batch size
        exe_drop = torch.tensor(np.random.rand(1)).to(device=self.device) >= 1-self.p_exe
        if not exe_drop:
            return feat_list, torch.ones([B], dtype=torch.float32, device=self.device)

        d_sum = sum(self.dim_list.values())
        q_sum = sum(self.dim_list[m] * q[m] for m in modality_list)
        theta = q_sum/d_sum
        num_mod = len(modality_list)
        q_temp = torch.tensor([q[m] for m in modality_list], device=self.device)
        mask = torch.distributions.Bernoulli(1 - q_temp).sample([B, 1]).permute(2, 1, 0).contiguous().reshape(num_mod, B, -1).to(self.device)
        
        cleaned = {}
        for idx, modality in enumerate(modality_list):
            D = feat_list[modality].shape[1]  
            current_mask = mask[idx].expand(-1,D)
            cleaned_fea = torch.mul(feat_list[modality], current_mask)
            cleaned_fea = cleaned_fea / (1 - theta + 1e-5)
            cleaned[modality] = cleaned_fea
                
        mask = mask.squeeze(-1).transpose(0,1) # [B,num_mod]

        update_flag = torch.sum(mask,dim=1) > 0
        for modality in modality_list:
            cleaned[modality] = cleaned[modality][update_flag]   
        return cleaned,update_flag
    
    
def clip(a, b, c):
    if b<a:
        return a
    if c<b:
        return c
    return b

class OPMTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {},args = {}):
        super(OPMTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.p_exe = method_dict['p_exe']
        self.q_base = method_dict['q_base']
        self.modality_drop=Modality_drop(dim_list=args.model['modality_size'],p_exe=self.p_exe,device=args.model["device"])
        
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
                # torch.cuda.empty_cache()
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
    
    def training_step(self, model : BaseClassifierModel, batch, batch_idx):

        # TODO: make it simpler and easier to extend
        softmax = nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss()
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()
        label = batch['label']
        label = label.to(model.device)
        model(batch)
        model.Unimodality_Calculate()
        loss = {}
        modality_list = model.modalitys
        key = list(model.modalitys)

        # Modulation starts here !
        modality_nums = len(modality_list) 
        scores = {}
        ratios = {}
        coeffs = {}
        #Calculate the scores

        
        if self.modulation_starts <= self.current_epoch <= self.modulation_ends: # bug fixed
            for modality in modality_list:
                scores[modality] = sum([softmax(model.unimodal_result[modality])[i][label[i]] for i in range(model.unimodal_result['output'].shape[0])])
            ##Calculate the ratios
            for modality in modality_list:
                ratios[modality] = scores[modality]
                if modality_nums == 2:
                    for modality_another in modality_list:
                        if modality_another == modality: 
                            continue
                      
                        ratios[modality] /= (scores[modality_another]+ 1e-5)  # prevent OOM
                        ratios[modality] = tanh(relu(ratios[modality]-1))
                if modality_nums == 3:
                    temp_score = 0.0
                    for modality_another in modality_list:
                        if modality_another == modality: 
                            continue
                        temp_score += scores[modality_another]
                    ratios[modality] /= (temp_score + 1e-5)
                    ratios[modality] = tanh(relu(ratios[modality]-1))
            #Calculate the coeffs
            for modality in modality_list:
                coeffs[modality] = self.q_base * (1 + self.alpha * ratios[modality]) if ratios[modality]>0 else 0
                coeffs[modality] = clip(coeffs[modality],0.0,1.0)
            model.encoder_result.pop('output')
  
            cleaned_fea,update_flag=self.modality_drop.execute_drop(model.encoder_result,coeffs,model)
            
            model.unimodal_result['output'] = model.fusion_module(cleaned_fea)
            select_mask=update_flag!=0
            label=label[select_mask]
            
            
            for modality in modality_list:
              
                model.unimodal_result[modality]=model.unimodal_result[modality][select_mask]
                

            for modality in model.unimodal_result.keys():
                loss[modality] = criterion(model.unimodal_result[modality],label)
            
            loss['output'].backward()
            
        else:
            pass

        return loss['output']
