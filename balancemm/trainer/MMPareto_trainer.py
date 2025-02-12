from typing import Mapping
from torch.optim.optimizer import Optimizer as Optimizer
from .base_trainer import BaseTrainer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from balancemm.models.avclassify_model import BaseClassifierModel
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
import lightning as L
import torch
import re

class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0   
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))
    
    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( np.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = np.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, np.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        new_dps={}
        new_init_sol=[]

        for item in dps:
            new_dps[item]=dps[item].numpy()

        for item in init_sol:
            if(torch.is_tensor(item)):
                data=item.numpy()
            else:
                data=item
            new_init_sol.append(data)
        
        dps=new_dps
        init_sol=new_init_sol


        
        n=len(vecs)
        sol_vec = np.zeros(n)

        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]
                

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            if(bool(grads[t])):
                gn[t] = np.sqrt(np.sum([grads[t][gr].pow(2).sum().data.cpu() for gr in grads[t]]))
            else:
                continue
    elif normalization_type == 'loss':
        for t in grads:
            if(bool(grads[t])):
                gn[t] = losses[t]
            else:
                continue
    elif normalization_type == 'loss+':
        for t in grads:
            if(bool(grads[t])):
                gn[t] = losses[t] * np.sqrt(np.sum([grads[t][gr].pow(2).sum().data.cpu() for gr in grads[t]]))
            else:
                continue
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn
    
    
class MMParetoTrainer(BaseTrainer):
    
    
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(MMParetoTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.method = method_dict['method']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
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
        
        record_names = {}
        for modality in model.modalitys:
            record_names[modality] = []
        for name,param in  model.named_parameters():
            if 'head' in name: 
                continue
            for modality in model.modalitys:
                if modality in name:
                    record_names[modality].append((name,param))
                    continue
                
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
                self.training_step(model=model, batch=batch, batch_idx=batch_idx,record_names=record_names,optimizer = optimizer)
                optimizer.step()
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
    
    def training_step(self, model : BaseClassifierModel, batch, batch_idx,record_names,optimizer):

        # TODO: make it simpler and easier to extend
        softmax = nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss()
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()
        label = batch['label']
        label = label.to(model.device)
        model(batch)
        model.Unimodality_Calculate()
        modality_list = model.modalitys
        grads = {}
        for modality in modality_list:
            grads[modality] = {}


        if self.modulation_starts <= self.current_epoch <= self.modulation_ends: # bug fixed
            loss = {}
            for modality in model.unimodal_result.keys():
                loss[modality] = criterion(model.unimodal_result[modality],label)
                loss[modality].backward(retain_graph = True)
            for modality in modality_list:
                for loss_type in loss.keys():
                    if loss_type == modality:
                        for tensor_name, param in record_names[modality]:
                            if loss_type not in grads[modality].keys():
                                grads[modality][loss_type] = {}
                            grads[modality][loss_type][tensor_name] = param.grad.data.clone()
                        grads[modality][loss_type]["concat"] = torch.cat([grads[modality][loss_type][tensor_name].flatten()  for tensor_name, _ in record_names[modality]])
                    if loss_type == 'output':
                        for tensor_name, param in record_names[modality]:
                            if loss_type not in grads[modality].keys():
                                grads[modality][loss_type] = {}
                            grads[modality][loss_type][tensor_name] = param.grad.data.clone() 
                        grads[modality][loss_type]["concat"] = torch.cat([grads[modality][loss_type][tensor_name].flatten() for tensor_name, _ in record_names[modality]])
            
            optimizer.zero_grad()
            
            this_cos = {}
            for modality in modality_list:
                this_cos[modality]=F.cosine_similarity(grads[modality]['output']["concat"],grads[modality][modality]["concat"],dim=0)
                
            weight = {}
            for modality in modality_list:
                weight[modality] = [0,0]
            for modality in modality_list:
                if(this_cos[modality]>0):
                    weight[modality][0]=0.5
                    weight[modality][1]=0.5
                else:
                    modality, min_norm = MinNormSolver.find_min_norm_element([list(grads[modality][t].values()) for t in grads[modality].keys()])
                    
            total_loss = sum(loss.values())
            total_loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer = re.split('[_.]',str(name))
                    if('head' in layer):
                        continue
                    for modality in modality_list:
                        if modality in layer:
                            three_norm=torch.norm(param.grad.data.clone())
                            new_grad=2*weight[modality][0]*grads[modality]['output'][name]+2*weight[modality][1]*grads[modality][modality][name]
                            new_norm=torch.norm(new_grad)
                            diff=three_norm/new_norm
                            if(diff>1):
                                param.grad=diff*new_grad*self.alpha
                            else:
                                param.grad=new_grad*self.alpha

            total_loss = total_loss.item()
        else:
            pass


        return total_loss