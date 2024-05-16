import json
import os
import torch
import torch.utils.data as data
from pathlib import Path
from random import randrange
import numpy as np
import h5py
import pickle
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

import os
import pandas as pd
from PIL import Image
import re
import random
from torchvision import transforms

def find_classes(directory) :
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    return classes, class_to_idx,idx_to_class

# 1. Subclass torch.utils.data.Dataset
class UMPC_FoodDataset(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, args:dict,transform=None):
        # 3. Create class attributes
        targ_dir = "/data/users/zequn_yang/Food101"
        phase = args['mode']
        mode = 'all'
        resize = 384
        # resize = 224
        train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                                transforms.Resize((resize,resize)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                            #transforms.CenterCrop(resize),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.dataset_root = targ_dir
        self.csv_file_path = '%s/texts/%s_titles.csv' % (self.dataset_root, phase)
        self.img_dir='%s/images/%s' % (self.dataset_root, phase)
        
        self.data = pd.read_csv(self.csv_file_path)
        self.tokenizer = BertTokenizer.from_pretrained('/home/zequn_yang/Food_101_new/pretrained')
        # Setup transforms
        if phase == 'train': 
            self.transform = train_transforms
        else:
            self.transform = test_transforms
        if transform != None: self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx,self.idx_to_class = find_classes(self.img_dir)
        self.mode=mode
        print(phase)
    # 4. Make function to load images
    def load_image(self, index,img_path): #-> Image.Image:
        "Opens an image via a path and returns it."
        return Image.open(img_path).convert('RGB')
                
    def clean_text(self,raw_text):
        t = re.sub(r'^RT[\s]+', '', raw_text)# remove old style retweet text "RT"
        t = re.sub(r'https?:\/\/.*[\r\n]*', '', t)# remove hyperlinks
        t = re.sub(r'#', '', t) # remove hashtags
        return t
    
    def tokenize(self, sentence):
        ids = self.tokenizer(sentence ,
                             padding='max_length', max_length=40, truncation=True).items()
        return {k: torch.tensor(v) for k, v in ids}
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self):# -> int:
        "Returns the total number of samples."
        return len(self.data)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index): #returns  Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        sample=self.data.iloc[index]
        txt=sample['text'] 
        txt = self.clean_text(txt)
        text_tokens= self.tokenize(txt)
        
        class_name  = sample['label'] 
        class_idx = self.class_to_idx[class_name]
        
        if self.mode =="all":
            img_path = os.path.join(self.img_dir,sample['label'] ,sample["Image_path"] )
            img = self.load_image(index,img_path)
            # Transform if necessary
            if self.transform:
                # return {'visual':self.transform(img).unsqueeze(1),'audio':text_tokens['input_ids'],'text':txt, 'label':class_idx}
                return {'visual':self.transform(img),'audio':text_tokens,'text':txt, 'label':class_idx}
                return self.transform(img).unsqueeze(1), text_tokens, txt, class_idx 
            else:
                return img, text_tokens, txt, class_idx 
        elif self.mode =="Text_only":
            return  text_tokens, txt, class_idx            
        