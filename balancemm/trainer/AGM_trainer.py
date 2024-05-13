from typing import Mapping
from lightning import LightningModule
from torch.optim.optimizer import Optimizer as Optimizer
from .base_trainer import BaseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast

import lightning as L
import torch
from lightning_utilities import apply_to_collection
import math

class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, AVT, AV, AT, VT, A ,V ,T):
        return 1/3*(AVT -VT) + 1/6*(AV - V + AT - T) + 1/3*(A)


class Modality_Visual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, AVT, AV, VT, AT, A, V, T):
        return 1/3*(AVT -AT) + 1/6*(AV - A + VT - T) + 1/3*(V)

class Modality_Text(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, AVT, AT, VT, AV, A, V, T):
        return 1/3*(AVT -AV) + 1/6*(VT - V + AT - A) + 1/3*(T)
    
class Modality_Visual_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, total_out, pad_visual_out, pad_audio_out):
        return 0.5 * (total_out - pad_visual_out + pad_audio_out)


class Modality_Audio_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, total_out, pad_visual_out, pad_audio_out):
        return 0.5 * (total_out - pad_audio_out + pad_visual_out)

class GradMod(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.mode = cfgs.mode
        self.extract_mm_feature = False
        # if cfgs.fusion_type == 'late_fusion': #early
        #     self.net = model
        # elif cfgs.fusion_type == 'early_fusion':
        #     self.net = AV_Early_Classifier(cfgs)
        self.net = model
        self.m_v = Modality_Visual()
        self.m_a = Modality_Audio()
        self.m_t = Modality_Text()
        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()
        self.m_t_o = Modality_out()

        self.scale_a = 1.0
        self.scale_v = 1.0
        self.scale_t = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_v_o.register_full_backward_hook(self.hookv)
        self.m_t_o.register_full_backward_hook(self.hookt)

    def hooka(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,

    def hookt(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_t,

    def update_scale(self, coeff_a, coeff_v, coeff_t):
        self.scale_a = coeff_a
        self.scale_v = coeff_v
        self.scale_t = coeff_t

    def forward(self, batch):
        _, _, _, AVT = self.net(batch, pad_audio=False, pad_visual=False, pad_text = False)
        # print(f'2.1 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')
        self.net.eval()
        with torch.no_grad():
            _, _, _, AT = self.net(batch, pad_audio=False, pad_visual=True, pad_text = False)
            _, _, _, VT = self.net(batch, pad_audio=True, pad_visual=False, pad_text = False)
            _, _, _, AV = self.net(batch, pad_audio=False, pad_visual=False, pad_text = True)
            _, _, _, T = self.net(batch, pad_audio=True, pad_visual=True, pad_text = False)
            _, _, _, V = self.net(batch, pad_audio=True, pad_visual=False, pad_text = True)
            _, _, _, A = self.net(batch, pad_audio=False, pad_visual=True, pad_text = True)
        # print(f'2.2 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')

        # if self.mode == "train":
        self.net.train()
        m_a = self.m_a_o(self.m_a(AVT, AV, AT, VT, A, V ,T))
        m_v = self.m_v_o(self.m_v(AVT, AV, VT, AT, A, V, T))
        m_t = self.m_t_o(self.m_t(AVT, AT, VT, AV, A, V, T))
        # print(f'2.3 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')
        # c = total_out - pad_audio_out - pad_visual_out + zero_padding_out
        # if self.extract_mm_feature is True:
        #     return total_out, pad_visual_out, pad_audio_out, zero_padding_out, m_a + m_v, encoded_feature
        #     # return m_a+m_v, m_a, m_v, zero_padding_out, m_a + m_v, encoded_feature
        
        return m_a, m_v, m_t, m_a + m_v + m_t

class GradMod_2(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.mode = cfgs.mode
        self.mode = 'train'
        self.extract_mm_feature = False
        self.net = model
        self.m_v = Modality_Visual_2()
        self.m_a = Modality_Audio_2()
        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()

        self.scale_a = 1.0
        self.scale_v = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_v_o.register_full_backward_hook(self.hookv)

    def hooka(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,

    def update_scale(self, coeff_a, coeff_v):
        self.scale_a = coeff_a
        self.scale_v = coeff_v

    def forward(self, batch):
        _,_,total_out = self.net(batch, pad_audio=False, pad_visual=False)
        # print(f'2.1 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')
        self.net.eval()
        with torch.no_grad():
            _, _, pad_visual_out = self.net(batch, pad_audio=False, pad_visual=True)
            _, _, pad_audio_out = self.net(batch, pad_audio=True, pad_visual=False)
            _, _, zero_padding_out = self.net(batch, pad_audio=True, pad_visual=True)
        # print(f'2.2 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')

        if self.mode == "train":
            self.net.train()
        m_a = self.m_a_o(self.m_a(total_out, pad_visual_out, pad_audio_out))
        m_v = self.m_v_o(self.m_v(total_out, pad_visual_out, pad_audio_out))
        # print(f'2.3 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')

        if self.extract_mm_feature is True:
            return total_out, pad_visual_out, pad_audio_out, zero_padding_out, m_a + m_v, encoded_feature
        return m_a, m_v, _ ,m_a + m_v
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, any] | None:
        out_a, out_v, out_t , out = self(batch)
        n_classes = self.net.n_classes
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        label = label.to(self.net.device)
        loss = F.cross_entropy(out, label)
        prediction = softmax(out)
        pred_v = softmax(out_v)
        pred_a = softmax(out_a)
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]

        for i in range(label.shape[0]):

            ma = np.argmax(prediction[i].cpu().data.numpy())
            v = np.argmax(pred_v[i].cpu().data.numpy())
            a = np.argmax(pred_a[i].cpu().data.numpy())
            num[label[i]] += 1.0

            #pdb.set_trace()
            if np.asarray(label[i].cpu()) == ma:
                acc[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == v:
                acc_v[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == a:
                acc_a[label[i]] += 1.0

        return loss, sum(acc), sum(acc_a), sum(acc_v), sum(acc_t)

class AGMTrainer(BaseTrainer):
    def __init__(self,fabric, method_dict: dict = {}, para_dict : dict = {}):
        super(AGMTrainer,self).__init__(fabric,**para_dict)
        self.alpha = method_dict['alpha']
        self.modulation_starts = method_dict['modulation_starts']
        self.modulation_ends = method_dict['modulation_ends']
        self.modality = method_dict['modality']

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

        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )
        total_batch = len(train_loader)
        train_score_a, train_score_v, train_score_t = 0, 0, 0
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
                loss, train_score_a, train_score_v, train_score_t = self.training_step(model=model, batch=batch,\
                                                                             batch_idx=batch_idx, total_batch = total_batch,\
                                                                                train_score_a = train_score_a, train_score_t = train_score_t,\
                                                                                    train_score_v = train_score_v)
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
    
    def training_step(self, model, batch, batch_idx, total_batch, train_score_a, train_score_v, train_score_t):

        # TODO: make it simpler and easier to extend
        step = batch_idx

        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        if self.modality == 2:
            Mod = GradMod_2(model)
        else:
            Mod = GradMod(model)
        out_a, out_v, out_t, out = Mod(batch)
        label = batch['label']
        label = label.to(model.device)
        # print(a.shape, v.shape, model.head.weight.shape)

        ## our modality-wise normalization on weight and feature
    
        iteration = (self.current_epoch) * total_batch + step + 1
        #avg_batch = get_segment_wise_relation(label,cfgs)
        # out_a = 0.5*(total_out-pad_audio_out + pad_visual_out)
        # out_v = 0.5*(total_out - pad_visual_out +pad_audio_out)
        # out_a, out_v, out_t = model.AVTCalculate(a, v, t, out) 
        # print(f'2 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')
        # print(out.shape)
        loss = criterion(out, label)

        # loss = loss + loss_avs * cfgs.lam_AGM
        loss_a = criterion(out_a, label)
        loss_v = criterion(out_v, label)


        # # calculate acc
        # prediction = softmax(out)
        # pred_a = softmax(out_a)
        # pred_v = softmax(out_v)
        # for j in range(image.shape[0]):
        #     ma = np.argmax(prediction[j].cpu().data.numpy())
        #     v = np.argmax(pred_v[j].cpu().data.numpy())
        #     a = np.argmax(pred_a[j].cpu().data.numpy())
        #     num[label[j]] += 1.0

        #     if np.asarray(label[j].cpu()) == ma:
        #         acc[label[j]] += 1.0
        #     if np.asarray(label[j].cpu()) == v:
        #         acc_v[label[j]] += 1.0
        #     if np.asarray(label[j].cpu()) == a:
        #         acc_a[label[j]] += 1.0

        
        if torch.isnan(out_a).any() or torch.isnan(out_v).any():
            raise ValueError
        score_audio = 0.
        score_visual = 0.
        score_text = 0.
        for k in range(out_a.size(0)):
            if torch.isinf(torch.log(softmax(out_a)[k][label[k]])) or softmax(out_a)[k][label[k]] < 1e-8:
                score_audio += -torch.log(torch.tensor(1e-8, dtype=out_a.dtype, device=out_a.device))
            else:
                score_audio += -torch.log(softmax(out_a)[k][label[k]])

            if torch.isinf(torch.log(softmax(out_v)[k][label[k]])) or softmax(out_v)[k][label[k]] < 1e-8:
                score_visual += -torch.log(torch.tensor(1e-8, dtype=out_v.dtype, device=out_v.device))
            else:
                score_visual += -torch.log(softmax(out_v)[k][label[k]])

            if self.modality ==2:
                pass
            if torch.isinf(torch.log(softmax(out_t)[k][label[k]])) or softmax(out_t)[k][label[k]] < 1e-8:
                score_text += -torch.log(torch.tensor(1e-8, dtype=out_v.dtype, device=out_v.device))
            else:
                score_text += -torch.log(softmax(out_v)[k][label[k]])
        score_audio = score_audio / out_a.size(0)
        score_visual = score_visual / out_v.size(0)
        if self.modality == 3:
            score_text = score_text / out_t.size(0)

        
            score_mean = (score_audio + score_visual + score_text)/3
            ratio_a = math.exp(3/2*(score_audio.item() - score_mean.item())) ##r
            ratio_v = math.exp(3/2*(score_visual.item() - score_mean.item()))
            ratio_t = math.exp(3/2*(score_text.item() - score_mean.item()))

            train_score_mean = (train_score_a + train_score_v +train_score_v)/3
            optimal_ratio_a = math.exp(3/2*(train_score_a - train_score_mean)) ##eta
            optimal_ratio_v = math.exp(3/2*(train_score_v - train_score_mean))
            optimal_ratio_t = math.exp(3/2*(train_score_t - train_score_mean))


            # coeff_a = math.exp(cfgs.alpha * (optimal_ratio_a - ratio_a))
            # coeff_v = math.exp(cfgs.alpha * (optimal_ratio_v - ratio_v))

            # if optimal_ratio_a - ratio_a> 10 or optimal_ratio_v - ratio_v >10 or optimal_ratio_t - ratio_t >10:
            #     print('difference:',optimal_ratio_a - ratio_a,optimal_ratio_v - ratio_v)
            coeff_a = math.exp(self.alpha * min(optimal_ratio_a - ratio_a,10))
            coeff_v = math.exp(self.alpha * min(optimal_ratio_v - ratio_v,10))
            coeff_t = math.exp(self.alpha * min(optimal_ratio_t - ratio_t,10))

            train_score_a = train_score_a * (iteration - 1) / iteration + score_audio.item() / iteration ##s^
            train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
            train_score_t = train_score_t * (iteration - 1) / iteration + score_text.item() / iteration
            # ra_score_a = ra_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            # ra_score_v = ra_score_v * step / (step + 1) + score_visual.item() / (step + 1)

            # if cfgs.method == "AGM" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
            if self.modulation_starts <= self.current_epoch <= self.modulation_ends:  
            
                Mod.update_scale(coeff_a, coeff_v, coeff_t)
            # print(f'3 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')
        else:
            ratio_a = math.exp(score_visual.item() - score_audio.item()) ##r
            ratio_v = math.exp(score_audio.item() - score_visual.item())

            optimal_ratio_a = math.exp(train_score_v - train_score_a) ##eta
            optimal_ratio_v = math.exp(train_score_a - train_score_v)
            coeff_a = math.exp(self.alpha * min(optimal_ratio_a - ratio_a,10))
            coeff_v = math.exp(self.alpha * min(optimal_ratio_v - ratio_v,10))
            train_score_a = train_score_a * (iteration - 1) / iteration + score_audio.item() / iteration ##s^
            train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
            train_score_t = 0

            if self.modulation_starts <= self.current_epoch <= self.modulation_ends:  
            
                Mod.update_scale(coeff_a, coeff_v)

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        grad_max = torch.max(Mod.net.fusion_module.fc_out.weight.grad)
        grad_min = torch.min(Mod.net.fusion_module.fc_out.weight.grad)
        if grad_max > 1 or grad_min < -1:
            nn.utils.clip_grad_norm_(Mod.parameters(), max_norm=1.0)
        return loss, train_score_a, train_score_v, train_score_t
    
    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
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
        # if val_loader is not None and not is_overridden("validation_step", _unwrap_objects(model)):
        #     L.fabric.utilities.rank_zero_warn(
        #         "Your LightningModule does not have a validation_step implemented, "
        #         "but you passed a validation dataloder. Skipping Validation."
        #     )
        #     return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")

        count = 0
        _acc = 0
        if self.modality == 2:
            model = GradMod_2(model)
        else:
            model = GradMod(model)
        count = 0
        _acc = 0
        _acc_a = 0
        _acc_v = 0
        _acc_t = 0
        valid_loss = 0
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out, acc, acc_a, acc_v, acc_t = model.validation_step(batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")

            count += len(batch)
            _acc += acc
            _acc_a += acc_a
            _acc_v += acc_v
            _acc_t += acc_t
            valid_loss += out
        valid_loss /= count
        _acc /= count
        _acc_a /= count
        _acc_v /= count
        _acc_t /= count
        #print("valid_acc : {}".format(_acc))
        if _acc > self.best_acc:
            self.should_save = True
            self.best_acc = _acc
        print("valid_acc : {}".format(_acc))

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)
        return valid_loss, _acc, _acc_a, _acc_v, _acc_t
