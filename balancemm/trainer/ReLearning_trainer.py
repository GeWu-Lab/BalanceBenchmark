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
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from operator import mod
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from ..evaluation.precisions import BatchMetricsCalculator
from ..models.avclassify_model import BaseClassifierModel
from ..evaluation.complex import FLOPsMonitor
from ..evaluation.modalitys import Calculate_Shapley
from logging import Logger
import os

def purity_score(y_true, y_pred):
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)



class ReLearningTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {},args = {}):
        super(ReLearningTrainer,self).__init__(fabric,**para_dict)
        self.move_lambda = method_dict['move_lambda']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.reinit_epoch = method_dict['reinit_epoch']
        self.reinit_num = method_dict['reinit_num']
        self.init_params = {}
        
        # self.modality = method_dict['modality']

    def get_feature(self,model,data_loader):
        iterable = self.progbar_wrapper(
            data_loader, total=len(data_loader), desc=f"Start getting feature ... "
        )
        features = {}
        for modality in model.modalitys:
            features[modality] = []
        features['label'] = []
        model.eval()
        with torch.no_grad():
            
            for batch_idx, batch in enumerate(iterable):
                label = batch['label'].to(model.device)
                model(batch)
                for modality in model.modalitys:
                    features[modality].append(model.encoder_result[modality].data.cpu())
                features['label'].append(label)

            for modality in model.modalitys:
                features[modality] = torch.cat(features[modality])
            features['label'] = torch.cat(features['label'])
        return features
            
    def reinit_score(self,model,train_features,val_features):

        train_purity = {}
        val_purity = {}
        purity_gap = {}
        weight = {}
        train_features['label'] = train_features['label'].cpu().numpy()
        val_features['label'] = val_features['label'].cpu().numpy()
        for modality in model.modalitys:
            result = train_features[modality].cpu().numpy()
            scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
            result = scaler.fit_transform(result)
            y_pred = KMeans(n_clusters=model.n_classes, random_state=0).fit_predict(result)
            train_purity[modality] = purity_score(train_features['label'],y_pred)

        for modality in model.modalitys:
            result = val_features[modality].cpu().numpy()
            scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
            result = scaler.fit_transform(result)
            y_pred = KMeans(n_clusters=model.n_classes, random_state=0).fit_predict(result)
            val_purity[modality] = purity_score(val_features['label'],y_pred)
        
        for modality in model.modalitys:
            purity_gap[modality] = np.abs(train_purity[modality]-val_purity[modality])


        for modality in model.modalitys:
            weight[modality] = torch.tanh(torch.tensor(self.move_lambda*purity_gap[modality]))
            
        return weight
    
    
    def reinit(self, model,weight):
        print("Start reinit ... ")
    
        for name, param in model.named_parameters():
            for modality in model.modalitys:
                if modality in name:
                    init_weight=self.init_params[name]
                    current_weight=param.data
                    new_weight=weight[modality]*init_weight+(1-weight[modality]).to(model.device)*current_weight
                    param.data=new_weight

        
        return model

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
        flag_reinit=0
        for name, param in model.named_parameters():
            self.init_params[name] = param.data.clone()
        while not self.should_stop:
            if self.should_train:
                
                model.train()

                self.train_loop(
                    model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg
                )
                if((self.current_epoch % self.reinit_epoch == 0)&(self.current_epoch>0)):
                    flag_reinit+=1
                    if(flag_reinit<=self.reinit_num):
                        print('reinit %d' % flag_reinit)
                        print("Start getting training feature ... ")
                        model.eval()
                        train_features = self.get_feature(model,train_loader)
                        print("Start getting evluating feature ... ")
                        val_features = self.get_feature(model,val_loader)
                        weight= self.reinit_score(model,train_features,val_features)
                        
                        model=self.reinit(model,weight)
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
                self.precision_calculator.ClearAll()
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

        softmax = nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss()
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()
        label = batch['label']
        label = label.to(model.device)
        model(batch)
        # model.Unimodality_Calculate()
        modality_list = model.modalitys


        if self.modulation_starts <= self.current_epoch <= self.modulation_ends: # bug fixed
            loss = criterion(model.unimodal_result['output'],label)
            loss.backward()
        else:
            pass


        return loss
    
