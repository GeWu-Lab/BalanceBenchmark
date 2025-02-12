import csv
from genericpath import isdir
import os
import random
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


import csv
from genericpath import isdir
import os
import random
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class UCF101Dataset(Dataset):

    def __init__(self, args:dict, v_norm = True, a_norm = False, name = "UCF101"):
        self.data = []
        classes = []
        data2class = {}
        self.mode= args['mode']
        self.v_norm = v_norm
        self.a_norm = a_norm
        self.stat_path = args['stat_path']
        self.train_txt = args['train_txt']
        self.test_txt = args['test_txt']
        self.visual_path = args['visual_path']
        self.flow_path_v = args['flow_path_v']
        self.flow_path_u = args['flow_path_u']

        if self.mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(self.stat_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")[1]
                classes.append(item)
        with open(csv_file) as f:
            for line in f:
                class_name = line.split('/')[0]
                name = line.split('/')[1].split('.')[0]
                if os.path.isdir(self.visual_path + name) and os.path.isdir(self.flow_path_u + name) and os.path.isdir(self.flow_path_v + name):
                    self.data.append(name)
                    data2class[name] = class_name   
        self.classes = sorted(classes)
        self.data2class = data2class
        self.class_num = len(self.classes)
        print(self.class_num)
        print('# of files = %d ' % len(self.data))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datum = self.data[idx]
        # crop = transforms.RandomResizedCrop(112, (1/4, 1.0), (3/4, 4/3))
        if self.mode == 'train':
            rgb_transf = [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
            diff_transf = [transforms.ToTensor()]

            flow_transf = [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            rgb_transf = [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor()
            ]
            diff_transf = [transforms.ToTensor()]
            flow_transf = [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ]

        if self.v_norm:
            rgb_transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            diff_transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if self.a_norm :
            flow_transf.append(transforms.Normalize([0.1307], [0.3081]))
        rgb_transf = transforms.Compose(rgb_transf)
        diff_transf = transforms.Compose(diff_transf)
        flow_transf = transforms.Compose(flow_transf)
        folder_path = self.visual_path + datum

        ####### RGB
        file_num = 6
        
        pick_num = 3
        seg = int(file_num/pick_num)
        image_arr = []

        for i in range(pick_num):
            if self.mode == 'train':
                chosen_index = random.randint(i*seg + 1, i*seg + seg)
            else:
                chosen_index = i*seg + max(int(seg/2), 1)
            path = folder_path + '/frame_0000' + str(chosen_index) + '.jpg'
            tranf_image = rgb_transf(Image.open(path).convert('RGB'))
            image_arr.append(tranf_image.unsqueeze(0))
        
        images = torch.cat(image_arr)

        num_u = len(os.listdir(self.flow_path_u + datum))
        pick_num = 3
        flow_arr = []
        seg = int(num_u/pick_num)

        for i in range(pick_num):
            if self.mode == 'train':
                chosen_index = random.randint(i*seg + 1, i*seg + seg)
            else:
                chosen_index = i*seg + max(int(seg/2), 1)

            flow_u = self.flow_path_u + datum + '/frame00' + str(chosen_index).zfill(4) + '.jpg'
            flow_v = self.flow_path_v + datum + '/frame00' + str(chosen_index).zfill(4) + '.jpg'
            u = flow_transf(Image.open(flow_u))
            v = flow_transf(Image.open(flow_v))
            flow = torch.cat((u,v),0)
            flow_arr.append(flow.unsqueeze(0))

            flow_n = torch.cat(flow_arr)
        images = images.permute(1,0,2,3)
        sample = {
            'flow':flow_n,
            'visual':images,
            'label': self.classes.index(self.data2class[datum]),
            'raw':datum,
            'idx':idx
        }


        return sample

