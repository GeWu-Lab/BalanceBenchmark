import csv
import os
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os.path as osp

class CremadDataset(Dataset):
    def __init__(self, args: dict):
        
        #  mode: str, # 'train', 'val', 'test'
        #  visual_path: str,
        #  audio_path: str,
        #  csv_file: str,
        #  fps: int = 3
        self.image = []
        self.audio = []
        self.label = []
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.mode = args['mode']
        self.fps = args['fps']
        self.visual_path = args['visual_path']
        self.audio_path = args['audio_path']
        # self.csv_file = args['csv_file']
        self.train_txt = args['train_txt']
        self.test_txt = args['test_txt']
        # self.visual_path = '/data/users/public/cremad/visual/'
        # self.audio_path = '/data/users/public/cremad/audio/'
        # self.stat_path = '/data/users/public/cremad/stat.csv'
        # self.train_txt = '/data/users/public/cremad/train.csv'
        # self.test_txt = '/data/users/public/cremad/test.csv'
        self.aid_transform = transforms.Compose([transforms.ToTensor()])
        if self.mode == 'train':
            self.csv_file = self.train_txt
        else:
            self.csv_file = self.test_txt
        self.data = []
        self.label = []
        with open(self.csv_file) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if item[1] in class_dict and os.path.exists(osp.join(self.audio_path, item[0] + '.pt')) and os.path.exists(osp.join(self.visual_path, item[0])) and len(os.listdir(osp.join(self.visual_path, item[0]))) >= self.fps:
                    self.data.append(item[0])
                    self.label.append(class_dict[item[1]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]

        # Audio
        fbank = torch.load(self.audio_path + datum + '.pt').unsqueeze(0)

        # Visual
        if self.mode == 'train':
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        folder_path = self.visual_path + datum
        file_num = len(os.listdir(folder_path))
        file_list = os.listdir(folder_path)
        pick_num = self.fps
        seg = int(file_num/pick_num)
        image_arr = []

        for i in range(pick_num):
            if self.mode == 'train':
                index = i*seg + np.random.randint(seg)
            else:
                index = i*seg + seg//2
            path = os.path.join(folder_path, file_list[index])
            image_arr.append(transf(Image.open(path).convert('RGB')).unsqueeze(0))

        images = torch.cat(image_arr)

        label = self.label[idx]
        images = images.permute(1,0,2,3)
        return {'audio': fbank, 'visual': images, 'label': label}

if __name__ == '__main__':   
    print('start')
    a = CremadDataset({'mode':'train','fps':2})
    a.__getitem__(0)