import csv
import math
import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset
from torchvision import transforms
import glob




class ModelNet40Dataset(Dataset):

    def __init__(self, args:dict):     
        self.mode=args['mode'] 
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = '/data/users/yake_wei/ModelNet40/modelnet40_images_new_12x/'+self.mode

        self.num_views = 12
        self.specific_view = [0,6]
        
        if(self.mode=='train'):
            self.transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        else:
            self.transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        ending='/*.png'
        self.init_filepaths(ending)

    def init_filepaths(self, ending):
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob('/data/users/yake_wei/ModelNet40/modelnet40_images_new_12x'+'/'+self.classnames[i]+'/'+self.root_dir.split('/')[-1]+ending))
            files = [] 
            for file in all_files:
                files.append(file.split('.obj.')[0])
                
            files = list(np.unique(np.array(files)))
            self.filepaths.extend(files)

        print('data load finish')

        print('# of files = %d ' % len(self.filepaths))


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
            path = self.filepaths[idx]
            class_name = path.split('/')[-3]
            class_id = self.classnames.index(class_name)
            imgs = torch.load(path+'.obj.npy')
            trans_imgs = []
            for img, view in zip(imgs[self.specific_view], self.specific_view):
                if self.transform:
                    img = self.transform(img)
                trans_imgs.append(img) 
            data = torch.stack(trans_imgs)
            return {"front_view":trans_imgs[0],"back_view":trans_imgs[1], "label": class_id, "idx": idx}