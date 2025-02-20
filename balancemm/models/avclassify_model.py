import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
# from .resnet_arch import ResNet18, ResNet
from .fusion_arch import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_3, ConcatFusion_N,SharedHead
# from typing import Mapping
# import numpy as np
# from .encoders import image_encoder, text_encoder
from ..encoders import create_encoders
import torch.distributed as dist
# def build_encoders(config_dict: dict[str, str])->dict[str, nn.Module]:
#     modalitys = config_dict.keys()
#     for modality in modalitys:
#         encoder_class = find_encoder(modality, config_dict[modality]['name'])
#         config_dict[modality] = encoder_class(config_dict[modality])
#     return config_dict
import warnings
class MultiModalParallel(nn.DataParallel):
    """self-difined DataParallel, support multi-GPUs validation"""
    def __init__(self, model, device_ids):
        super().__init__(model, device_ids)
    @property
    def n_classes(self):
        return self.module.n_classes
        
    @property
    def fusion(self):
        return self.module.fusion
        
    @property
    def modalitys(self):
        return self.module.modalitys
        
    @property
    def enconders(self):
        return self.module.enconders
        
    @property
    def modality_encoder(self):
        return self.module.modality_encoder
        
    @property
    def device(self):
        return self.module.device
        
    @property
    def modality_size(self):
        return self.module.modality_size
        
    @property
    def encoder_result(self):
        return self.module.encoder_result
        
    @encoder_result.setter
    def encoder_result(self, value):
        self.module.encoder_result = value
        
    @property
    def unimodal_result(self):
        return self.module.unimodal_result
        
    @unimodal_result.setter
    def unimodal_result(self, value):
        self.module.unimodal_result = value
        
    @property
    def prediction(self):
        return self.module.prediction
        
    @prediction.setter
    def prediction(self, value):
        self.module.prediction = value

    @property
    def fusion_module(self):
        return self.module.fusion_module
    def validation_step(self, batch, batch_idx, limit_modality):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Was asked to gather along dimension 0"
            )
            return self.forward(batch, 
                            batch_idx=batch_idx, 
                            limit_modality=limit_modality, 
                            _run_validation=True)  
    def unimodality_calculate(self, mask = None, dependent_modality = {}):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Was asked to gather along dimension 0"
            )
            return self.forward((mask, dependent_modality), _run_Unimodality=True)  
    def feature_extract(self,batch,**kwargs):
        replicas = self.replicate(self.module, self.device_ids)
        batch_scattered, kwargs_scattered = self.scatter(batch, kwargs, self.device_ids)
        results = self.parallel_apply(
                [replica.feature_extract for replica in replicas],
                batch_scattered, kwargs_scattered
            )
        encoder_result = [r for r in results]
        modality_res = self.gather(encoder_result, self.output_device)
        return modality_res
    def forward_grad(self,batch, **kwargs):
        replicas = self.replicate(self.module, self.device_ids)
        batch_scattered, kwargs_scattered = self.scatter(batch, kwargs, self.device_ids)
        results = self.parallel_apply(
                [replica.forward_grad for replica in replicas],
                batch_scattered, kwargs_scattered
            )
        encoder_result = [r[0] for r in results]
        unimodal_result = [r[1] for r in results]
        prediction = [r[2] for r in results]
        self.module.encoder_result = self.gather(encoder_result, self.output_device)
        self.module.unimodal_result = self.gather(unimodal_result, self.output_device)
        self.module.prediction = self.gather(prediction, self.output_device)
        return self.module.unimodal_result['output']
    def modality_model(self, batch, **kwargs):
        replicas = self.replicate(self.module, self.device_ids)
        batch_scattered, kwargs_scattered = self.scatter(batch, kwargs, self.device_ids)
        results = self.parallel_apply(
            [replica.modality_model for replica in replicas],
            batch_scattered,
            kwargs_scattered
        )
        
        encoder_result = [r for r in results]
        modality_res = self.gather(encoder_result, self.output_device)
        return modality_res

    def forward(self, batch, **kwargs):
        if kwargs.pop('_run_validation', False):
           
            replicas = self.replicate(self.module, self.device_ids)
            kwargs_None = None
            batch_scattered, kwargs_scattered = self.scatter(batch, kwargs, self.device_ids)
        
            results = self.parallel_apply(
                [replica.validation_step for replica in replicas],
                batch_scattered, kwargs_scattered
            )
            
            outputs = [r[0] for r in results]
            encoder_result = [r[1] for r in results]
            unimodal_result = [r[2] for r in results]
            prediction = [r[3] for r in results]
            self.module.encoder_result = self.gather(encoder_result, self.output_device)
            self.module.unimodal_result = self.gather(unimodal_result, self.output_device)
            self.module.prediction = self.gather(prediction, self.output_device)
            
            return self.gather(outputs, self.output_device).sum()
        elif kwargs.pop('_run_Unimodality', False):
            replicas = self.replicate(self.module, self.device_ids)
            mask_scattered, kwargs_scattered = self.scatter(batch, kwargs, self.device_ids)            # kwargs_scattered = self.scatter(kwargs, target_gpus=self.device_ids)
            results = self.parallel_apply(
                [replica.unimodality_calculate for replica in replicas],
                mask_scattered,
                kwargs_scattered
            )
            encoder_result = [r[0] for r in results]
            unimodal_result = [r[1] for r in results]
            prediction = [r[2] for r in results]
            self.module.encoder_result = self.gather(encoder_result, self.output_device)
            self.module.unimodal_result = self.gather(unimodal_result, self.output_device)
            self.module.prediction = self.gather(prediction, self.output_device)
                
            
            return self.module.unimodal_result
        else:
            replicas = self.replicate(self.module, self.device_ids)
            
            batch_scattered, kwargs_scattered = self.scatter(batch, kwargs, self.device_ids)
            results = self.parallel_apply(
                replicas,
                batch_scattered, kwargs_scattered
            )
            encoder_result = [r[0] for r in results]
            unimodal_result = [r[1] for r in results]
            prediction = [r[2] for r in results]
            self.module.encoder_result = self.gather(encoder_result, self.output_device)
            self.module.unimodal_result = self.gather(unimodal_result, self.output_device)
            self.module.prediction = self.gather(prediction, self.output_device)
           
            return self.module.encoder_result
        
class BaseClassifierModel(nn.Module):
    # mid fusion
    def __init__(self, args):
        super(BaseClassifierModel, self).__init__()
        self.n_classes = args['n_classes']
        self.fusion = args['fusion']
        self.modalitys = args['encoders'].keys()
        self.enconders = args['encoders']
        self.modality_encoder = nn.ModuleDict(create_encoders(args['encoders']))
        self.device = 'cuda:0' if args['device'] != ' ' else 'cpu'
        self.modality_size = args['modality_size']
        self.encoder_result = {}
        self.unimodal_result = {}
        self.prediction = {}
        if self.fusion == 'sum':
            self.fusion_module = SumFusion(output_dim = self.n_classes)
        elif self.fusion == 'concat':
            self.fusion_module = ConcatFusion_N(input_dim = sum(self.modality_size.values()) ,output_dim = self.n_classes)
        elif self.fusion == 'film':
            self.fusion_module = FiLM(output_dim = self.n_classes, x_film=True)
        elif self.fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim = self.n_classes, x_gate=True)
        elif self.fusion == 'shared':
            self.fusion_module = SharedHead(input_dim = max(self.modality_size.values()),output_dim = self.n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion))
        
    def resnet_process(self, modality_data : torch.Tensor, modality : str) -> torch.Tensor:
        B = len(modality_data)
        if modality == 'visual' or modality == 'flow' or modality == 'front_view' or modality == 'back_view': 
            if modality == 'visual':
                modality_data = modality_data.permute(0, 2, 1, 3, 4).contiguous().float()
            else:
                modality_data = modality_data.contiguous().float()
            result = self.modality_encoder[modality](modality_data)
            (_, C, H, W) = result.size()
            result = result.view(B, -1, C, H, W)
            result = result.permute(0, 2, 1, 3, 4)
            result = F.adaptive_avg_pool3d(result, 1)
            result = torch.flatten(result, 1)
        elif modality == 'audio':
            result = self.modality_encoder[modality](modality_data)
            result = F.adaptive_avg_pool2d(result, 1)
            result = torch.flatten(result, 1)
        
        return result
    
    def transformer_process(self, modality_data: torch.Tensor, modality: str)-> torch.Tensor:
        result = self.modality_encoder[modality](modality_data)
        return result
    
    def vit_process(self, modality_data: torch.Tensor, modality: str)-> torch.Tensor:
        modality_data = modality_data.unsqueeze(1)
        result = self.modality_encoder[modality](modality_data)
        return result

    def encoder_process(self, modality_data : torch.Tensor, modality_name: str) -> torch.Tensor:
        
        encoder_name = self.enconders[modality_name]['name']
        if encoder_name == 'ResNet18':
            result = self.resnet(modality_data = modality_data, modality = modality_name)
        elif encoder_name == 'Transformer' or encoder_name == 'Transformer_':
            result = self.transformer_process(modality_data = modality_data, modality = modality_name)
        elif encoder_name == 'ViT_B':
            result = self.vit_process(modality_data = modality_data, modality = modality_name)
        elif encoder_name == 'Transformer_LA':
            result = self.transformer_process(modality_data = modality_data, modality = modality_name)
        return result
    
    def forward(self,
                batch,
                padding = [],
                mask = None, 
                dependent_modality = {"audio": False, "visual": False, "text": False}, 
                pt = 0,
                types= 0) -> dict[str, torch.Tensor]:
        self.encoder_result = {}
        for modality in self.modalitys:
            modality_data = batch[modality]
            # modality_data = modality_data.to(self.device)
            if modality in padding:
                if mask is None:
                    modality_data = torch.zeros_like(modality_data, device=modality_data.device)
            modality_res = self.encoder_process(modality_data = modality_data, modality_name= modality)
            self.encoder_result[modality] = modality_res 
        self.encoder_result['output'] = self.fusion_module(self.encoder_result)
        ##new
        self.unimodality_calculate()
        return self.encoder_result,self.unimodal_result, self.prediction
    
    def unimodality_calculate(self) -> dict[str, torch.Tensor]:
        modality_nums = 0
        all_nums = len(self.encoder_result.keys())-1
        self.unimodal_result = {}
        self.prediction = {}
        now_size = 0
        for modality in self.encoder_result.keys():
            if modality == 'output':
                self.unimodal_result[modality] = self.encoder_result[modality]
                continue
            if self.fusion == 'concat':
                weight_size = self.fusion_module.fc_out.weight.size(1)
                self.unimodal_result[modality] = (torch.mm(self.encoder_result[modality],\
                                               torch.transpose(self.fusion_module.fc_out.weight[:,\
                                                                                                now_size :\
                                                                                                now_size + self.modality_size[modality]], 0, 1))
                                    + self.fusion_module.fc_out.bias / all_nums)
                now_size += self.modality_size[modality]
            modality_nums += 1
        
        softmax =nn.Softmax(dim= 1)
        for modality in self.unimodal_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        return self.encoder_result,self.unimodal_result, self.prediction

    def validation_step(self, batch : dict[str, torch.Tensor], batch_idx : int, limit_modality: list) -> tuple[torch.Tensor, dict[str, list]]:
    
        self.prediction = {}
        padding = []
        for modality in self.modalitys:
            if modality not in limit_modality:
                padding.append(modality)
        self(batch, padding = padding)
        self.unimodality_calculate()
        n_classes = self.n_classes
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        # label = label.to(self.device)
        out = self.unimodal_result['output']
        # print(out.device, label.device)
        loss = F.cross_entropy(out, label)
        loss = loss * len(label)
        for modality in self.unimodal_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        return loss,self.encoder_result,self.unimodal_result, self.prediction


class BaseClassifierAMCoModel(BaseClassifierModel):
    def __init__(self, args):
        super(BaseClassifierAMCoModel, self).__init__(args)
        self.linear_star = nn.Linear(in_features=self.n_classes, out_features= self.n_classes)
    def forward(self,
                batch,
                padding = [],
                mask = None, 
                dependent_modality = {}, 
                pt = 0,
                types= 0) -> dict[str, torch.Tensor]:
        self.encoder_result = {}
        self.unimodal_result = {}
        self.prediction = {}
        for modality in self.modalitys:
            modality_data = batch[modality]
            # modality_data = modality_data.to(self.device)
            if modality in padding:
                modality_data = torch.zeros_like(modality_data, device=modality_data.device)
                modality_res = self.encoder_process(modality_data = modality_data, modality_name= modality)
            elif dependent_modality.get(modality, False) == True:
                modality_res = self.encoder_process(modality_data = modality_data, modality_name= modality)
            else :
                modality_res = self.encoder_process(modality_data = modality_data, modality_name= modality)
            self.encoder_result[modality] = modality_res 
        self.encoder_result['output'] = self.fusion_module(self.encoder_result)
        self.unimodality_calculate(mask,dependent_modality)
        self.encoder_result['output'] = self.linear_star(self.encoder_result['output'])
        self.unimodal_result['output'] = self.encoder_result['output'] 
        return self.encoder_result, self.unimodal_result, self.prediction
    
    def unimodality_calculate(self, mask= None, dependent_modality = {}) -> dict[str, torch.Tensor]:
        softmax = nn.Softmax(dim = 1)
        modality_nums = 0
        all_nums = len(self.encoder_result.keys())-1
        self.unimodal_result = {}
        self.prediction = {} 
        self.unimodal_result['output'] = torch.zeros_like(self.encoder_result['output'])
        now_size = 0
        for modality in self.encoder_result.keys():
            if modality == 'output':
                continue
            if self.fusion == 'concat':
                weight_size = self.fusion_module.fc_out.weight.size(1)
                self.unimodal_result[modality] = (torch.mm(self.encoder_result[modality],\
                                               torch.transpose(self.fusion_module.fc_out.weight[:,\
                                                                                                now_size :\
                                                                                                now_size + self.modality_size[modality]], 0, 1))
                                    + self.fusion_module.fc_out.bias / all_nums)
                now_size += self.modality_size[modality]
            if dependent_modality.get(modality, False):
                self.unimodal_result[modality] = torch.mul(self.unimodal_result[modality], mask)
            self.unimodal_result['output'] += self.unimodal_result[modality]
            modality_nums += 1
        self.encoder_result['output'] = self.unimodal_result['output']
        for modality in self.unimodal_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        return self.encoder_result,self.unimodal_result, self.prediction


class BaseClassifierGreedyModel(BaseClassifierModel):
    def __init__(self, args):
        super(BaseClassifierGreedyModel, self).__init__(args)
        if 'text' in self.modalitys:
            self.mmtm_layers = nn.ModuleDict({
            "mmtm1": MMTM([40, 64], 4, self),
            "mmtm2": MMTM([40, 128], 4,self),
            "mmtm3": MMTM([40, 256], 4,self),
            "mmtm4": MMTM([40, 512], 4,self)
        })
        else:
            self.mmtm_layers = nn.ModuleDict({
                "mmtm1": MMTM([64, 64], 4, self),
                "mmtm2": MMTM([128, 128], 4, self),
                "mmtm3": MMTM([256, 256], 4, self),
                "mmtm4": MMTM([512, 512], 4, self)
            })
    def resnet_process(self, modality_data : torch.Tensor, modality : str) -> torch.Tensor:
        encoder = self.modality_encoder[modality]
        if modality == 'visual' or modality == 'flow' or modality == 'front_view' or modality == 'back_view': 
            if modality == 'visual':
                modality_data = modality_data.permute(0, 2, 1, 3, 4).contiguous().float()
            else:
                modality_data = modality_data.contiguous().float()
            # (B, T, C, H, W) = modality_data.size()
            # print(modality_data.size())
            # x = modality_data.view(B * T, C, H, W)
            (_, C, H, W)  = modality_data.size()
            x = modality_data.view(-1, C, H, W)
            x = encoder.conv1(x)
            x = encoder.bn1(x)
            x = encoder.relu(x)
            x = encoder.maxpool(x)
            x = encoder.layer1(x)
            
        elif modality == 'audio':
            x = modality_data
            x = encoder.conv1(x)
            x = encoder.bn1(x)
            x = encoder.relu(x)
            x = encoder.maxpool(x)
            x = encoder.layer1(x)
        
        return x
    
    def transformer_process(self, modality_data: torch.Tensor, modality: str)-> torch.Tensor:
        encoder = self.modality_encoder[modality]
        x = modality_data
        if self.enconders[modality]['if_pretrain'] == True:
            x = x.squeeze(1).int()
            outputs = encoder.textEncoder(x,output_hidden_states=True)
            hidden_states = outputs.hidden_states
            result = hidden_states[9]

        return result 

    def encoder_process(self, modality_data : torch.Tensor, modality_name: str) -> torch.Tensor:
        ## May be it could use getattr
        encoder_name = self.enconders[modality_name]['name']
        if encoder_name == 'ResNet18':
            result = self.resnet_process(modality_data = modality_data, modality = modality_name)
        elif encoder_name == 'Transformer':
            result = self.transformer_process(modality_data = modality_data, modality = modality_name)
        return result
    
        
    def forward(self,
                batch,
                curation_mode = False,
                caring_modality = 0,
                padding = [],
                mask = None, 
                dependent_modality = {}, 
                pt = 0,
                types= 0) -> dict[str, torch.Tensor]:
        self.encoder_result = {}
        for modality in self.modalitys:
            modality_data = batch[modality].to(self.device)
            if modality in padding:
                if mask is None:
                    modality_data = torch.zeros_like(modality_data, device=modality_data.device)
            modality_res = self.encoder_process(modality_data = modality_data, modality_name= modality)
            self.encoder_result[modality] = modality_res 
        
        key = list(self.modalitys)
        self.encoder_result = self.mmtm_layers["mmtm1"](
                self.encoder_result, self, curation_mode, caring_modality
            )
        modality_list = self.modalitys
        for i in [2,3,4]:
            for modality in modality_list:
                if modality == 'text':
                    self.encoder_result[modality] = self.modality_encoder[modality].textEncoder.encoder.layer[i+7](self.encoder_result[modality])[0]
                else:    
                    self.encoder_result[modality] = getattr(self.modality_encoder[modality],f'layer{i}')(self.encoder_result[modality])
            self.encoder_result = self.mmtm_layers[f"mmtm{i}"](
                self.encoder_result, self, curation_mode, caring_modality
            )
        for modality in modality_list:
            B = len(batch[modality])
            if self.enconders[modality]['name'] == 'ResNet18':
                if modality == 'visual' or modality == 'flow' or modality == 'front_view' or modality == 'back_view': 
                    result = self.encoder_result[modality]
                    (_, C, H, W) = result.size()
                    result = result.view(B, -1, C, H, W)
                    result = result.permute(0, 2, 1, 3, 4)
                    result = F.adaptive_avg_pool3d(result, 1)
                    result = torch.flatten(result, 1)
                    self.encoder_result[modality] = result
                elif modality == 'audio':
                    result = self.encoder_result[modality]
                    result = F.adaptive_avg_pool2d(result, 1)
                    result = torch.flatten(result, 1)
                    self.encoder_result[modality] = result
            else:
                result = self.encoder_result[modality]
                result = self.modality_encoder[modality].textEncoder.pooler(result)
                result = self.modality_encoder[modality].linear(result)
                self.encoder_result[modality] = result
        self.encoder_result['output'] = self.fusion_module(self.encoder_result)
        self.unimodality_calculate()
        return self.encoder_result
    
    def unimodality_calculate(self) -> dict[str, torch.Tensor]:
        modality_nums = 0
        all_nums = len(self.encoder_result.keys())-1
        self.unimodal_result = {}
        self.prediction = {}
        now_size = 0
        for modality in self.encoder_result.keys():
            if modality == 'output':
                self.unimodal_result[modality] = self.encoder_result[modality]
                continue
            if self.fusion == 'concat':
                weight_size = self.fusion_module.fc_out.weight.size(1)
                self.unimodal_result[modality] = (torch.mm(self.encoder_result[modality],\
                                               torch.transpose(self.fusion_module.fc_out.weight[:,\
                                                                                                now_size :\
                                                                                                now_size + self.modality_size[modality]], 0, 1))
                                    + self.fusion_module.fc_out.bias / all_nums)
                now_size += self.modality_size[modality]
            modality_nums += 1
            ##new
        softmax =nn.Softmax(dim= 1)
        for modality in self.unimodal_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        return self.unimodal_result
        
 
class MMTM(nn.Module):
    def __init__(self, modalities_dims, ratio,model:BaseClassifierGreedyModel):
        super(MMTM,self).__init__()
        self.modalities_dims = modalities_dims
        total_dim = sum(modalities_dims)
        dim_out = int(2*total_dim/ratio)
        
        self.fc_squeeze = nn.Linear(total_dim, dim_out)
        key = list(model.modalitys)
        self.fc_excites = nn.ModuleDict({key[0]:nn.Linear(dim_out,modalities_dims[0]),
                                         key[1]:nn.Linear(dim_out,modalities_dims[1])})
        self.running_avg_weight = {}
        i = 0
        for modality in model.modalitys:
            # self.fc_excites[modality] = nn.Linear(dim_out, modalities_dims[i]).to(f"cuda:{model.device}")
            self.running_avg_weight[modality] = torch.zeros(modalities_dims[i]).to(f"{model.device}")
            i+=1
        # self.fc_excites = nn.ModuleList([nn.Linear(dim_out, dim) for dim in modalities_dims])
        # self.running_avg_weights = torch.zeros(total_dim).to(self.device)
        # self.running_avg_weight_visual = torch.zeros(dim_visual).to("cuda:{}".format(device))
        # self.running_avg_weight_skeleton = torch.zeros(dim_visual).to("cuda:{}".format(device))
        self.step = 0
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self
                ,data
                ,model:BaseClassifierGreedyModel
                ,curation_mode = False
                ,caring_modality = 0):
        squeeze_array = []
        self.running_avg_weight = {}
        modality_list = model.modalitys
        for modality in modality_list:
            if modality == 'visual' or modality == 'flow':
                (_, C, H, W) = data[modality].size()
                # B = data['audio'].size()[0]
                B = 64
                data[modality] = data[modality].view(B,-1,C,H,W)
                data[modality] = data[modality].permute(0,2,1,3,4)
        
        for modality in modality_list:
            B,C = data[modality].shape[0],data[modality].shape[1]
            tview = data[modality].reshape(B,C,-1)
            squeeze_array.append(torch.mean(tview,dim=-1))
        # for tensor in [x,y]:
        #     tview = tensor.view(tensor.shape[:2] + (-1,))
        #     print(torch.mean(tview,dim=-1).shape)
        #     squeeze_array.append(torch.mean(tview,dim=-1))
        
        squeeze = torch.cat(squeeze_array, 1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        out = {}
        for modality in modality_list:
            out[modality] = self.fc_excites[modality](excitation)
            out[modality] = self.sigmoid(out[modality])
            self.running_avg_weight[modality] = (out[modality].mean(0) + self.running_avg_weight[modality]*self.step).detach()/(self.step+1)        
        self.step +=1
        dim_diff = {}
        key= list(modality_list)
        if not curation_mode:
            for modality in modality_list:
                dim_diff = len(data[modality].shape) - len(out[modality].shape)
                out[modality] = out[modality].view(out[modality].shape + (1,) * dim_diff)
        else:
            if caring_modality==0:
                dim_diff = len(data[key[0]].shape) - len(out[key[0]].shape)
                out[key[0]] = torch.stack(out[key[0]].shape[0]*[
                        self.running_avg_weight[key[0]]
                    ]).view(out[key[0]].shape + (1,) * dim_diff)
                dim_diff = len(data[key[1]].shape) - len(out[key[1]].shape)
                out[key[1]] = out[key[1]].view(out[key[1]].shape + (1,) * dim_diff)
                
            elif caring_modality==1:
                dim_diff = len(data[key[1]].shape) - len(out[key[1]].shape)
                out[key[1]] = torch.stack(out[key[1]].shape[0] * [
                        self.running_avg_weight[key[1]]
                    ]).view(out[key[1]].shape + (1,) * dim_diff)
                dim_diff = len(data[key[0]].shape) - len(out[key[0]].shape)
                out[key[0]] = out[key[0]].view(out[key[0]].shape + (1,) * dim_diff)
        for modality in modality_list:
            data[modality] = data[modality] * out[modality]
            if modality == "visual" or modality == 'flow':
                data[modality] = data[modality].permute(0,2,1,3,4).view(-1,C,H,W)
            if modality == 'text':
                data[modality] = data[modality].squeeze(2)
        return data

class BaseClassifierMLAModel(BaseClassifierModel):
    def __init__(self, args):
        super(BaseClassifierMLAModel, self).__init__(args)
        
    def forward(self,
                batch,
                padding = [],
                mask = None, 
                dependent_modality = {"audio": False, "visual": False, "text": False}, 
                pt = 0,
                types= 0) -> dict[str, torch.Tensor]:
        self.encoder_result = {}
        for modality in self.modalitys:
            modality_data = batch[modality]
            # modality_data = modality_data.to(self.device)
            if modality in padding:
                if mask is None:
                    modality_data = torch.zeros_like(modality_data, device=modality_data.device)
            modality_res = self.encoder_process(modality_data = modality_data, modality_name= modality)
            self.encoder_result[modality] = modality_res 
    
        # self.unimodality_calculate()
        return self.encoder_result, self.unimodal_result, self.prediction
    
    def unimodality_calculate(self) -> dict[str, torch.Tensor]:
        self.unimodal_result = {}
        self.prediction = {}
        for modality in self.encoder_result.keys():
            if self.fusion == 'shared':
                self.unimodal_result[modality] = self.fusion_module.fc_out(self.encoder_result[modality])
        key = list(self.modalitys)
        conf = self.calculate_gating_weights(self.unimodal_result)
        
        if len(self.modalitys) == 3:
            self.unimodal_result['output'] = self.unimodal_result[key[0]] * conf[key[0]] + self.unimodal_result[key[1]] * conf[key[1]] + self.unimodal_result[key[2]] * conf[key[2]]
        else:
            self.unimodal_result['output'] = self.unimodal_result[key[0]] * conf[key[0]] + self.unimodal_result[key[1]] * conf[key[1]] 
        # if len(self.modalitys) == 3:
        #     self.unimodal_result['output'] = self.unimodal_result[key[0]] * conf[key[0]] + self.unimodal_result[key[1]] * conf[key[1]] + self.unimodal_result[key[2]] * conf[key[2]]
        # else:
        #     self.unimodal_result['output'] = self.unimodal_result[key[0]] * av_alpha + self.unimodal_result[key[1]] * (1-av_alpha)
        # softmax =nn.Softmax(dim= 1)
        # for modality in self.unimodal_result.keys():
        #     softmax_res = softmax(self.unimodal_result[modality])
        #     self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        return self.encoder_result, self.unimodal_result, self.prediction

    def calculate_entropy(self,output):
        
        probabilities = F.softmax(output,dim=1)
        log_probabilities = torch.log(probabilities)
        entropy = -torch.sum(probabilities*log_probabilities,dim = 1,keepdim=True)
        
        return entropy

    def calculate_gating_weights(self,output):
        key = list(self.modalitys)
        entropy = {}
        gating_weight = {}
        for modality in self.modalitys:
            entropy[modality] = self.calculate_entropy(output[modality])
        max_entropy = torch.max(entropy[key[0]],entropy[key[1]])
        for modality in self.modalitys:
            gating_weight[modality] = torch.exp(max_entropy - entropy[modality])
        sum_weights = sum(gating_weight.values())
        # gating_weight_1 = torch.exp(max_entropy - entropy_1)
        # gating_weight_2 = torch.exp(max_entropy - entropy_2)
        
        # sum_weights = gating_weight + gating_weight_2
        for modality in self.modalitys:
            gating_weight[modality] = gating_weight[modality] / sum_weights  
        # gating_weight_1 /= sum_weights
        # gating_weight_2 /= sum_weights
        
        return gating_weight

    def validation_step(self, batch : dict[str, torch.Tensor], batch_idx : int, limit_modality: list) -> tuple[torch.Tensor, dict[str, list]]:
        # ** drop
        padding = []
        for modality in self.modalitys:
            if modality not in limit_modality:
                padding.append(modality)
        self(batch,padding=padding)
        self.unimodality_calculate()
        n_classes = self.n_classes
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        # label = label.to(self.device)
        out = self.unimodal_result['output']
        loss = F.cross_entropy(out, label)
        loss = loss * len(label)
        for modality in self.unimodal_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        # for modality in self.unimodal_result.keys():
        #     acc_res[modality] = [0.0 for _ in range(n_classes)]
        #     pred_res[modality] = softmax(self.unimodal_result[modality])
        # for i in range(label.shape[0]):
        #     for modality in self.unimodal_result.keys():
        #         modality_pred = np.argmax(pred_res[modality][i].cpu().data.numpy())
        #         if np.asarray(label[i].cpu()) == modality_pred:
        #             acc_res[modality][label[i]] += 1.0
            
        #     num[label[i]] += 1.0
        return loss,self.encoder_result,self.unimodal_result, self.prediction
    def feature_extract(self,batch,modality):
        modality_data = batch[modality]
        modality_res = self.encoder_process(modality_data = modality_data, modality_name= modality)
        return modality_res


class BaseClassifierReconBoostModel(BaseClassifierModel):
    # mid fusion
    def __init__(self, args):
        super(BaseClassifierReconBoostModel, self).__init__(args)
       
        self.boost_rate  = nn.Parameter(torch.tensor(0.01, requires_grad=True, device=self.device))
        
    def modality_model(self,batch,modality):
        modality_data = batch[modality]
        modality_res = self.encoder_process(modality_data = modality_data, modality_name= modality)
        modality_res = self.fusion_module.fc_out(modality_res)
        
        return modality_res
           
    def forward(self,
                batch,
                mask_model = None,
                padding = [],
                mask = None, 
                dependent_modality = {"audio": False, "visual": False, "text": False}, 
                pt = 0,
                types= 0) -> dict[str, torch.Tensor]:
        self.unimodal_result = {}
        self.prediction = {}
        self.unimodal_result['output'] = None
        with torch.no_grad():
            for modality in self.modalitys:
                if modality == mask_model:
                    continue
                modality_data = batch[modality]
                # modality_data = modality_data.to(self.device)
                modality_data = modality_data
                if modality in padding:
                    if mask is None:
                        modality_data = torch.zeros_like(modality_data, device=modality_data.device)
                self.unimodal_result[modality] = self.modality_model(batch,modality) ###
                if self.unimodal_result['output'] is None:
                    self.unimodal_result['output'] = self.unimodal_result[modality]
                else: 
                    self.unimodal_result['output'] = self.unimodal_result['output'] + self.unimodal_result[modality]
        self.unimodal_result['output'] = self.boost_rate * self.unimodal_result['output']
        softmax  = nn.Softmax(dim = 1)
        for modality in self.unimodal_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        return self.encoder_result, self.unimodal_result, self.prediction
    
    def forward_grad(self,
                batch,
                padding = [],
                mask = None, 
                dependent_modality = {"audio": False, "visual": False, "text": False}, 
                pt = 0,
                types= 0) -> dict[str, torch.Tensor]:
        self.unimodal_result = {}
        self.prediction = {}
        self.unimodal_result['output'] = None
        for modality in self.modalitys:
            modality_data = batch[modality]
            # modality_data = modality_data.to(self.device)
            modality_data = modality_data
            self.unimodal_result[modality] = self.modality_model(batch,modality)
            if self.unimodal_result['output'] is None:
                self.unimodal_result['output'] = self.unimodal_result[modality]
            else: 
                self.unimodal_result['output'] = self.unimodal_result['output'] + self.unimodal_result[modality]

        self.unimodal_result['output'] = self.boost_rate * self.unimodal_result['output']
        softmax  = nn.Softmax(dim = 1)
        for modality in self.unimodal_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
        return self.encoder_result, self.unimodal_result, self.prediction
    
    def validation_step(self, batch : dict[str, torch.Tensor], batch_idx : int, limit_modality: list) -> tuple[torch.Tensor, dict[str, list]]:
        # ** drop
        padding = []
        for modality in self.modalitys:
            if modality not in limit_modality:
                padding.append(modality)
        self.forward(batch, padding = padding)
        n_classes = self.n_classes
        softmax  = nn.Softmax(dim = 1)
        label = batch['label']
        # label = label.to(self.device)
        loss = F.cross_entropy(self.unimodal_result['output'], label)
        loss = loss * len(label)
        for modality in self.unimodal_result.keys():
            softmax_res = softmax(self.unimodal_result[modality])
            self.prediction[modality] = torch.argmax(softmax_res, dim = 1)
 
        return loss,self.encoder_result, self.unimodal_result, self.prediction
