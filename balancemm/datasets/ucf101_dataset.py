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


class UCFDataset(Dataset):

    def __init__(self, args: dict, transforms=None):
        self.data = []
        classes = []
        data2class = {}
        self.mode= args['mode']
   
        self.stat_path = '/data/users/zequn_yang/UCF-101/classInd.txt'
        self.train_txt = '/data/users/zequn_yang/UCF-101/trainlist01.txt'
        self.test_txt = '/data/users/zequn_yang/UCF-101/testlist01.txt'
        self.visual_path = '/data/users/zequn_yang/UCF-101/ucf101-frames-1fps/video-set/'
        self.flow_path_v = '/data/users/zequn_yang/UCF-101/v/'
        self.flow_path_u = '/data/users/zequn_yang/UCF-101/u/'

        if self.mode == 'train':
            csv_file = self.train_txt

        # elif mode == 'val':
        #     csv_file = self.val_txt
        #     self.visual_path = '/home/zequn_yang/vggsound/train/'

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
        # print(self.class_num)
        # print('# of files = %d ' % len(self.data))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        if self.mode == 'train':
            rgb_transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            diff_transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            flow_transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.1307], [0.3081])
            ])
        else:
            rgb_transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            diff_transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            flow_transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.1307], [0.3081])
            ])

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
            flow_v = self.flow_path_u + datum + '/frame00' + str(chosen_index).zfill(4) + '.jpg'
            u = flow_transf(Image.open(flow_u))
            v = flow_transf(Image.open(flow_v))
            flow = torch.cat((u,v),0)
            flow_arr.append(flow.unsqueeze(0))

            flow_n = torch.cat(flow_arr)
        return  {'flow':flow_n, 'visual':images,   'label':self.classes.index(self.data2class[datum])}

if __name__ == '__main__':   
    print('start')
    a = UCF_Dataset({'mode':'train'})
    a.__getitem__(0)
