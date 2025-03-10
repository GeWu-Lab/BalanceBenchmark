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

def clip(a, b, c):
    if b<a:
        return a
    if c<b:
        return c
    return b 

# quicker
# def EU_dist(x1, x2):
#     return torch.cdist(x1, x2, p=2)
def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix

class PMRTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(PMRTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.momentum_coef = method_dict['momentum_coef']
        self.eta = method_dict['eta']
        self.proto = {}
        
    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
    
        modality_list = model.modalitys
        all_modalitys = list(model.modalitys)
        all_modalitys.append('output')
        self.precision_calculator = self.PrecisionCalculatorType(model.n_classes, all_modalitys)
        
        self.fabric.call("on_train_epoch_start")
        if self.current_epoch == 0: 
            for modality in modality_list:
                self.proto[modality] = 0  

        self.proto = self.calculate_prototype(model, train_loader, proto0=self.proto)
        model.train()
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
                optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx, proto = self.proto))
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
    
    def training_step(self, model, batch, batch_idx, proto):

        # TODO: make it simpler and easier to extend
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)
        tanh = nn.Tanh()
        modality_list = model.modalitys
        key = list(modality_list)
        m = {}
        model(batch)
        for modality in modality_list:
            m[modality] = model.encoder_result[modality]
        # a, v = model(batch)['audio'], model(batch)['visual']
        unimodal_result = model.Unimodality_Calculate()
        # out_v,out_a,out = unimodal_result['visual'], unimodal_result['audio'], unimodal_result['output']
        label = batch['label']
        label = label.to(model.device)
        loss_modality = {}
        for modality in modality_list:
            # print(unimodal_result[modality])
            # print(label)
            loss_modality[modality] = criterion(unimodal_result[modality],label)

        if  self.modulation_starts <= self.current_epoch <= self.modulation_ends:
            sim = {}
            for modality in modality_list:
                sim[modality] = -EU_dist(m[modality],proto[modality])
            # audio_sim = -EU_dist(a, audio_proto)  # B x n_class
            # visual_sim = -EU_dist(v, visual_proto)  # B x n_class
            # print('sim: ', audio_sim[0][0].data, visual_sim[0][0].data, a[0][0].data, v[0][0].data)

            score_p = {}
            # score = {}
            loss_proto = {}
            for modality in modality_list:
                score_p[modality] = sum([softmax(sim[modality])[i][label[i]] for i in range(sim[modality].size(0))])

            # score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
            # score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
            if len(modality_list) == 2:
                ratio_a_p = score_p[key[0]] / score_p[key[1]]
            else:
                ratio = {}
                min_score = min(score_p.values())
                for modality in modality_list:
                    ratio[modality] = score_p[modality] / min_score

            # for modality in modality_list:
            #     score[modality] = sum([softmax(unimodal_result[modality])[i][label[i]] for i in range(unimodal_result[modality].size(0))])
            # # score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            # # score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
            # ratio_a = score[key[0]] / score[key[1]]

            for modality in modality_list:
                loss_proto[modality] = criterion(sim[modality],label)
            # loss_proto_a = criterion(audio_sim, label)
            # loss_proto_v = criterion(visual_sim, label)
            if len(modality_list) == 2: 
                if ratio_a_p > 1:
                    beta = 0  # audio coef
                    lam = 1 * self.alpha  # visual coef
                elif ratio_a_p < 1:
                    beta = 1 * self.alpha
                    lam = 0
                else:
                    beta = 0
                    lam = 0
                loss = criterion(unimodal_result['output'], label) + beta * loss_proto[key[0]] + lam * loss_proto[key[1]]
                loss.backward()
            else:
                loss = criterion(unimodal_result['output'], label)
                loss.backward()
                k_t = {}
                for modality in modality_list:
                    if ratio[modality] > 1:
                        k_t[modality] = 1-tanh(self.eta * ratio[modality])
                    else:
                        k_t[modality] = 1
                
                for name, parms in model.named_parameters():
                    layer = str(name)
                    for modality in modality_list:
                        if modality in layer and len(parms.grad.size()) != 1: ##Don't change the grad of bias for layer
                            parms.grad = parms.grad * k_t[modality]  - \
                                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            # loss_a = criterion(out_a, label)
        else:
            loss = criterion(unimodal_result['output'], label)
            loss.backward()
        # # avoid gradients in stored/accumulated values -> prevents potential OOM
        # self._current_train_return = apply_to_collection(model.encoder_result, dtype=torch.Tensor, function=lambda x: x.detach())
        return loss
    def calculate_prototype(self, model, dataloader, proto0):
    # todo customed output of prototype
        n_classes = model.n_classes
        device = next(model.parameters()).device
        proto = {}
        modality_list = model.modalitys
        for modality in modality_list:
            proto[modality] = torch.zeros(n_classes, model.modality_size[modality]).to(device)
        count_class = [0 for _ in range(n_classes)]

        # calculate prototype
        model.eval()
        with torch.no_grad():
            sample_count = 0
            all_num = len(dataloader)
            m = {}
            for batch_idx, batch in enumerate(dataloader):
                model(batch)
                for modality in modality_list:
                    m[modality] = model.encoder_result[modality]
                label = batch['label']


                for c, l in enumerate(label):
                    l = l.long()
                    count_class[l] += 1
                    for modality in modality_list:
                    
                        proto[modality][l,:] += m[modality][c,:]


                sample_count += 1

                if sample_count >= all_num // 10:
                    break
            for modality in modality_list:
                for c in range(proto[modality].shape[0]):
                    proto[modality][c,:] /= count_class[c]
                # audio_prototypes[c, :] /= count_class[c]
                # visual_prototypes[c, :] /= count_class[c]

            if self.current_epoch <= 0:
                for modality in modality_list:
                        proto[modality] = proto[modality]
                # audio_prototypes = audio_prototypes
                # visual_prototypes = visual_prototypes
            else:
                for modality in modality_list:
                        proto[modality] = (1-self.momentum_coef) * proto[modality] + self.momentum_coef * proto0[modality]
                # audio_prototypes = (1 - self.momentum_coef) * audio_prototypes + self.momentum_coef * a_proto
                # visual_prototypes = (1 - self.momentum_coef) * visual_prototypes + self.momentum_coef * v_proto
            return proto