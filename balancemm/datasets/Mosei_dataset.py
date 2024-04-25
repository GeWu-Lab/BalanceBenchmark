import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch
from torchvision import transforms

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class MoseiDataset(Dataset):
    def __init__(self, args: dict, transforms = None):
        super(MoseiDataset, self).__init__()
        # dataset_path = '/home/zequn_yang/MMT/data/'
        dataset_path = args['dataset_path']
        # data='mosei_senti'
        data= args['data']
        split_type= args['mode']
        if_align = args['if_align']
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))
        # split_type =  'train'# args['mode']

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        # self.audio = self.audio.cuda()
        # self.text = self.text.cuda()
        # self.vision = self.vision.cuda()
        # self.labels = self.labels.cuda()
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data

        # if split_type == 'train':
        #     self.transf = transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
        # else:
        #     self.transf = transforms.Compose([
        #         transforms.Resize(size=(224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
        
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = [index, self.text[index], self.audio[index], self.vision[index]]
        # X = torch.tensor(X).cpu()
        Y = self.labels[index]
        Y = (Y +3).round()
        Y = int(Y.item())
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        # for indx in range(1000):
        #     print(self.vision[indx].shape)
        return {'text' : X[1], 'visual': X[3], 'audio' : X[2], 'label':Y}
        return self.text[index], Y   ##, META        
