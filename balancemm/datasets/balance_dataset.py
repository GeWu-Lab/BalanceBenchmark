import json
import h5py
import os
import pickle
from PIL import Image
import io
import torch
import csv
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import random
import copy

class BalanceDataset(Dataset):

    def __init__(self, args: dict, transforms=None):
        self.data = []
        self.label = []
        self.mode = None
        csv_path = args['csv_path']
        self.visual_path = args['visual_path']
        self.audio_path = args['audio_path']
        
        self.mode = args['mode']


        with open(csv_path) as f:
            annotation_data = json.load(f)
            all_data = annotation_data['database']
            # choose = ['playing piano', 'playing cello', 'lawn mowing', 'singing', 'cleaning floor', 'bowling', 'swimming', 'whistling', 'motorcycling', 'playing flute', 'writing on blackboard', 'beat boxing']
            class_labels = annotation_data['labels']

        self.class_to_idx = {label : i for i,label in enumerate(class_labels)}
        print(len(class_labels))
        # exit(0)

        for key in all_data.keys():
        #     print(all_data[key])
        #   exit(0)
        
            if all_data[key]['subset'] == (self.mode + 'ing'):
                if os.path.exists(self.visual_path + key + '.hdf5') and os.path.exists(self.audio_path + key + '.pkl'):
                    self.data.append(key)
                    self.label.append(self.class_to_idx[all_data[key]['label']])


        print('data load finish')

        self.transforms = transforms

        self._init_atransform()

        print('# of files = %d ' % len(self.data))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        with open(self.audio_path + av_file + '.pkl',"rb") as f:
          spectrogram = pickle.load(f)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # Visual
        path = self.visual_path + av_file + '.hdf5'
        with h5py.File(path, 'r') as f:
          video_data = f['video']
          file_num = len(video_data)

          if self.mode == 'training':

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

          pick_num = 3
          seg = int(file_num / pick_num)
          image = []
          image_arr = []
          t = [0] * pick_num

          for i in range(pick_num):
              if self.mode == 'train':
                  t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                  if t[i] >= 9:
                      t[i] = 8
              else:
                  t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1

              image.append(Image.open(io.BytesIO(video_data[t[i]])).convert('RGB'))

              image_arr.append(transf(image[i]))
              image_arr[i] = image_arr[i].unsqueeze(1).float()
              if i == 0:
                  image_n = copy.copy(image_arr[i])
              else:
                  image_n = torch.cat((image_n, image_arr[i]), 1)
        

        label = self.label[idx]

        return {'visual':image_n, 'audio':spectrogram, 'label': label}
        # return  image_n,spectrogram,label,idx