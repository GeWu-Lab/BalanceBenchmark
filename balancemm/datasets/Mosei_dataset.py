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

def acc3(i):
    if i<-0.5:
        return 0
    if i>0.5:
        return 1
    return 2

def acc7(i):
    if i < -2:
        res = 0
    if -2 <= i and i < -1:
        res = 1
    if -1 <= i and i < 0:
        res = 2
    if 0 <= i and i <= 0:
        res = 3
    if 0 < i and i <= 1:
        res = 4
    if 1 < i and i <= 2:
        res = 5
    if i > 2:
        res = 6
    return res

def acc2(i):
    if i<0:
        return 0
    else :
        return 1

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
        # Y = torch.where(Y>0 , 1, 0) 2classes
        # Y = int(Y.item())
        # new_Y = torch.zeros(1, 1)
        # new_Y[0, 0] = acc7(Y[0,0])
        Y = acc2(Y[0,0])
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        # for indx in range(1000):
        #     print(self.vision[indx].shape)
        return {'text' : X[1], 'visual': X[3], 'audio' : X[2], 'label':Y}
        return self.text[index], Y   ##, META        

if __name__ == '__main__':
    dataset_path = '/home/zequn_yang/MMT/data/'
    data='mosei_senti'
    split_type= 'valid'
    #train = 16326 test = 4659
    if_align = False
    dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
    dataset = pickle.load(open(dataset_path, 'rb'))
    count = {0:0, 1:0}
    for split_type in dataset.keys():
        count = {0:0, 1:0}
        for i in range(len(dataset[split_type]['labels'])):
            Y = acc2(dataset[split_type]['labels'][i][0][0])
            count[Y]+=1
        print(split_type, count)