import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .resnet_arch import ResNet18, ResNet
from .fusion_arch import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_3
from typing import Mapping
import numpy as np
class AVClassifierModel(L.LightningModule):
    def __init__(self, args):
        super(AVClassifierModel, self).__init__()
        n_classes = args['n_classes']
        fusion = args['fusion']

        self.n_classes = n_classes
        self.audio_net = ResNet18(modality='audio')
        self.visual_net = ResNet18(modality='visual')

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))


    def forward(self, audio, visual):
        visual = visual.permute(0, 2, 1, 3, 4).contiguous().float()
        audio = audio.unsqueeze(1).float()
        # print(a.shape)
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out
    
    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, any] | None:
        modality_1, modality_2 = batch[1], batch[0]
        a, v, out = self(modality_1, modality_2)
        loss = F.cross_entropy(out, batch[2])
        return loss
    
class Transformer(nn.Module):
    """
    Extend to nn.Transformer.
    """
    def __init__(self,input_dim = 300, n_features = 512,dim = 1024,n_head = 4,n_layers = 2):
        super(Transformer,self).__init__()
        self.embedding = nn.Linear(input_dim, n_features)
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features,self.embed_dim,kernel_size=1,padding=0,bias=False)
        layer = TransformerEncoder(self.embed_dim,num_heads=n_head, dim_feedforward = n_features)
        self.transformer = nn.TransformerEncoder(layer,num_layers=n_layers)


    def forward(self,x):
        """
        Apply transorformer to input tensor.

        """
        if type(x) is list:
            x = x[0]
        # x = x.to(self.embedding.weight.dtype)
        x = self.embedding(x)
        x = self.conv(x.permute([0,2,1]))
        x = x.permute([2,0,1])
        x = self.transformer(x)[0]
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src,is_causal, src_mask=None, src_key_padding_mask=None):
        # 自注意力层
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class AVTClassifierModel(nn.Module):
    def __init__(self, args):
        super(AVTClassifierModel, self).__init__()
        n_classes = args['n_classes']
        fusion = args['fusion']

        self.n_classes = n_classes
        # self.audio_net = ResNet18(modality='audio')
        # self.visual_net = ResNet18(modality='visual')
        self.audio_net = Transformer(input_dim = 74)
        self.visual_net = Transformer(input_dim = 35)
        self.text_net = Transformer()
        self.fc_a = nn.Linear(1024,1)
        self.fc_v = nn.Linear(1024,1)
        self.fc_t = nn.Linear(1024,1)
        self.fusion = fusion
        self.device = None
        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion_3(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))


    def forward(self,
                batch,
                pad_audio = False,
                pad_visual = False, 
                pad_text = False, 
                mask = None, 
                dependent_modality = {"audio": False, "visual": False, "text": False}, 
                pt = 0,
                types= 0):
        visual = batch['visual']
        audio = batch['audio']
        text = batch['text']
        print(visual.shape)
        print(audio)
        print(text)
        visual = visual.to(self.device)
        audio = audio.to(self.device)
        text = text.to(self.device)
        if pad_audio:
            audio = torch.zeros_like(audio, device=audio.device)
        if pad_visual:
            visual = torch.zeros_like(visual, device=visual.device)
        if pad_text:
            text = torch.zeros_like(text, device=text.device)
        # visual = visual.permute(0, 2, 1, 3, 4).contiguous().float()
        # audio = audio.unsqueeze(1).float()
        # print(a.shape)
        if types == 1:
            a = self.audio_net(audio)
            a = torch.flatten(a, 1)
            out_a = self.fc_a(a)
            return out_a
        
        if types == 2:
            v = self.visual_net(audio)
            v = torch.flatten(v, 1)
            out_v = self.fc_v(v)
            return out_v
        
        if types == 3:
            t = self.audio_net(t)
            t = torch.flatten(t, 1)
            out_t = self.fc_t(t)
            return out_t
        a = self.audio_net(audio)
        v = self.visual_net(visual)
        t = self.text_net(text)
        # (_, C, H, W) = v.size()
        # B = a.size()[0]
        # v = v.view(B, -1, C, H, W)
        # v = v.permute(0, 2, 1, 3, 4)

        # a = F.adaptive_avg_pool2d(a, 1)
        # v = F.adaptive_avg_pool2d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        t = torch.flatten(t, 1)

        if dependent_modality['audio']:
            a = torch.mul(a,mask)
            if(abs(pt-1)>0.1):
                a = a*1/(1-pt)
            else:
                a = a*10
        elif dependent_modality['visual']:
            v = torch.mul(v,mask)
            if(abs(pt-1)>0.1):
                v = v*1/(1-pt)
            else:
                v = v*10
        elif dependent_modality['text']:
            t = torch.mul(t, mask)
            if(abs(pt-1)>0.1):
                t = t*1/(1-pt)
            else:
                t = t*10

        a, v, t, out = self.fusion_module(a, v, t)
        return a, v, t, out
    
    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, any] | None:
        a, v, t,out = self(batch)
        loss = F.cross_entropy(out, batch['label'])
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, any] | None:
        
        a, v, t , out = self(batch)
        out_a, out_v, out_t = self.AVTCalculate(a, v ,t , out)
        n_classes = self.n_classes
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        label = label.to(self.device)
        loss = F.cross_entropy(out, label)
        prediction = softmax(out)
        pred_v = softmax(out_v)
        pred_a = softmax(out_a)
        pred_t = softmax(out_t)
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]

        for i in range(label.shape[0]):

            ma = np.argmax(prediction[i].cpu().data.numpy())
            v = np.argmax(pred_v[i].cpu().data.numpy())
            a = np.argmax(pred_a[i].cpu().data.numpy())
            t = np.argmax(pred_t[i].cpu().data.numpy())
            num[label[i]] += 1.0

            #pdb.set_trace()
            if np.asarray(label[i].cpu()) == ma:
                acc[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == v:
                acc_v[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == a:
                acc_a[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == t:
                acc_t[label[i]] += 1.0
        return loss, sum(acc), sum(acc_a), sum(acc_v), sum(acc_t)
    
    def AVCalculate(self, a, v, out):
        if self.fusion == 'sum':
            out_v = (torch.mm(v, torch.transpose(self.fusion_module.fc_y.weight, 0, 1)) +
                        self.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(self.fusion_module.fc_x.weight, 0, 1)) +
                        self.fusion_module.fc_x.bias)
        elif self.fusion == 'concat':
            weight_size = self.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(self.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                        + self.fusion_module.fc_out.bias / 2)
            out_a = (torch.mm(a, torch.transpose(self.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                        + self.fusion_module.fc_out.bias / 2)
        elif self.fusion == 'film':
            out_v = out
            out_a = out
        elif self.fusion == 'gated':
            out_v = out
            out_a = out
        return out_a, out_v
    
    def AVTCalculate(self, a, v, t, out):
        if self.fusion == 'sum':
            pass
        elif self.fusion == 'concat':
            weight_size = self.fusion_module.fc_out.weight.size(1)
            out_a = (torch.mm(a, torch.transpose(self.fusion_module.fc_out.weight[:, 0: weight_size // 3], 0, 1))
                     + self.fusion_module.fc_out.bias / 3)
            out_v = (torch.mm(v, torch.transpose(self.fusion_module.fc_out.weight[:, weight_size // 3: 2 * weight_size // 3], 0, 1))
                     + self.fusion_module.fc_out.bias / 3)
            out_t = (torch.mm(t, torch.transpose(self.fusion_module.fc_out.weight[:, 2* weight_size // 3: weight_size ], 0, 1))
                    + self.fusion_module.fc_out.bias / 3)
        elif self.fusion == 'film':
            out_v = out
            out_a = out
            out_t = out
        elif self.fusion == 'gated':
            out_v = out
            out_a = out
            out_t = out
        return out_a, out_v, out_t
    
class AVTClassifier_gbModel(AVTClassifierModel):
    def __init__(self, args):
        super(AVTClassifier_gbModel, self).__init__(args)
        n_classes = args['n_classes']
        fusion = args['fusion']

        self.n_classes = n_classes
        # self.audio_net = ResNet18(modality='audio')
        # self.visual_net = ResNet18(modality='visual')
        self.audio_net = Transformer(input_dim = 74)
        self.visual_net = Transformer(input_dim = 35)
        self.text_net = Transformer()
        self.fc_a = nn.Linear(1024,n_classes)
        self.fc_v = nn.Linear(1024,n_classes)
        self.fc_t = nn.Linear(1024,n_classes)
        self.fusion = fusion
        self.device = None
        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion_3(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        
    def forward(self,
                batch,
                pad_audio = False,
                pad_visual = False, 
                pad_text = False, 
                mask = None, 
                dependent_modality = {"audio": False, "visual": False, "text": False}, 
                pt = 0,
                types= 0):
        visual = batch['visual']
        audio = batch['audio']
        text = batch['text']
        visual = visual.to(self.device)
        audio = audio.to(self.device)
        text = text.to(self.device)
        if pad_audio:
            audio = torch.zeros_like(audio, device=audio.device)
        if pad_visual:
            visual = torch.zeros_like(visual, device=visual.device)
        if pad_text:
            text = torch.zeros_like(text, device=text.device)
        # visual = visual.permute(0, 2, 1, 3, 4).contiguous().float()
        # audio = audio.unsqueeze(1).float()
        # print(a.shape)
        if types == 1:
            a = self.audio_net(audio)
            a = torch.flatten(a, 1)
            out_a = self.fc_a(a)
            return out_a
        
        if types == 2:
            v = self.visual_net(visual)
            v = torch.flatten(v, 1)
            out_v = self.fc_v(v)
            return out_v
        
        if types == 3:
            t = self.text_net(text)
            t = torch.flatten(t, 1)
            out_t = self.fc_t(t)
            return out_t
        a = self.audio_net(audio)
        v = self.visual_net(visual)
        t = self.text_net(text)
        # (_, C, H, W) = v.size()
        # B = a.size()[0]
        # v = v.view(B, -1, C, H, W)
        # v = v.permute(0, 2, 1, 3, 4)

        # a = F.adaptive_avg_pool2d(a, 1)
        # v = F.adaptive_avg_pool2d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        t = torch.flatten(t, 1)

        if dependent_modality['audio']:
            a = torch.mul(a,mask)
            if(abs(pt-1)>0.1):
                a = a*1/(1-pt)
            else:
                a = a*10
        elif dependent_modality['visual']:
            v = torch.mul(v,mask)
            if(abs(pt-1)>0.1):
                v = v*1/(1-pt)
            else:
                v = v*10
        elif dependent_modality['text']:
            t = torch.mul(t, mask)
            if(abs(pt-1)>0.1):
                t = t*1/(1-pt)
            else:
                t = t*10

        a, v, t, out = self.fusion_module(a, v, t)

        out_a = self.fc_a(a)
        out_v = self.fc_v(v)
        out_t = self.fc_t(t)
        return out_a, out_v, out_t, out
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, any] | None:
        
        out_a, out_v, out_t , out = self(batch)
        n_classes = self.n_classes
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        label = label.to(self.device)
        loss = F.cross_entropy(out, label)
        prediction = softmax(out)
        pred_v = softmax(out_v)
        pred_a = softmax(out_a)
        pred_t = softmax(out_t)
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]

        for i in range(label.shape[0]):

            ma = np.argmax(prediction[i].cpu().data.numpy())
            v = np.argmax(pred_v[i].cpu().data.numpy())
            a = np.argmax(pred_a[i].cpu().data.numpy())
            t = np.argmax(pred_t[i].cpu().data.numpy())
            num[label[i]] += 1.0

            #pdb.set_trace()
            if np.asarray(label[i].cpu()) == ma:
                acc[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == v:
                acc_v[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == a:
                acc_a[label[i]] += 1.0
            if np.asarray(label[i].cpu()) == t:
                acc_t[label[i]] += 1.0
        return loss, sum(acc), sum(acc_a), sum(acc_v), sum(acc_t)
    
class VTClassifierModel(AVTClassifierModel):
    def __init__(self, args):
        super(VTClassifierModel, self).__init__(args)
        #### audio means text
        n_classes = args['n_classes']
        fusion = args['fusion']

        self.n_classes = n_classes
        # self.audio_net = ResNet18(modality='audio')
        self.visual_net = ResNet18(modality='visual')
        # self.visual_net = Transformer(input_dim = 35)
        self.audio_net = Transformer(input_dim=40, dim=512)
        self.fusion = fusion
        self.device = None
        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes, input_dim=1024)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        
    def forward(self,
                batch,
                pad_audio = False,
                pad_visual = False, 
                pad_text = False, 
                mask = None, 
                dependent_modality = {"audio": False, "visual": False, "text": False}, 
                pt = 0,
                types= 0):
        visual = batch['visual']
        audio = batch['audio']
        visual = visual.to(self.device)
        audio = audio.to(self.device)
        if pad_audio:
            audio = torch.zeros_like(audio, device=audio.device)
        if pad_visual:
            visual = torch.zeros_like(visual, device=visual.device)
        visual = visual.permute(0, 2, 1, 3, 4).contiguous().float()
        audio = audio.unsqueeze(1).float()

        a = self.audio_net(audio)
        v = self.visual_net(visual)
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        # a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        if dependent_modality['audio']:
            a = torch.mul(a,mask)
            if(abs(pt-1)>0.1):
                a = a*1/(1-pt)
            else:
                a = a*10
        elif dependent_modality['visual']:
            v = torch.mul(v,mask)
            if(abs(pt-1)>0.1):
                v = v*1/(1-pt)
            else:
                v = v*10

        a, v, out = self.fusion_module(a, v)
        return a, v, out
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, any] | None:
        
        a, v,  out = self(batch)
        out_a, out_v = self.AVCalculate(a,v,out)
        n_classes = self.n_classes
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        label = label.to(self.device)
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