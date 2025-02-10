import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random

class VGGDataset(Dataset):

    def __init__(self, args, mode='train', csv_root='/data/users/shaoxuan_xu/data/vggsound.csv'):
        self.args = args
        self.mode = args['mode']
        train_video_data = []
        train_audio_data = []
        test_video_data  = []
        test_audio_data  = []
        train_label = []
        test_label  = []
        train_class = []
        test_class  = []

        video_train_root = '/data/users/public/vggsound/train'
        video_test_root  = '/data/users/public/vggsound/test'
        audio_train_root = '/data/users/public/VGGSound_Audio_npy/VGGSound_Audio_processed_and_saved_as_npy'
        audio_test_root  = '/data/users/public/VGGSound_Audio_npy/VGGSound_Audio_processed_and_saved_as_npy'

        print(video_train_root)
        print(video_test_root)
        print(audio_train_root)
        print(audio_test_root)
        train_valid = 0
        test_valid = 0
        with open(csv_root) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if item[3] == 'train':
                    # video_dir = os.path.join('/data/users/xiaokang_peng/VGGsound/', 'train-videos/train-set-img', 'Image-{:02d}-FPS'.format(self.args.fps), item[0]+'_'+item[1]+'.mp4')
                    # video_dir = os.path.join(dataroot, 'train/visual_all/visual', item[0]+'_'+item[1])
                    # audio_dir = os.path.join(dataroot, 'train/audio_np', item[0]+'_'+item[1] + '.npy')

                    video_dir = os.path.join(video_train_root, item[0]+'_'+item[1])
                    # audio_dir = os.path.join(audio_train_root, item[0]+'_'+item[1] + '.wav')
                    audio_dir = os.path.join(audio_train_root, item[0]+'_'+item[1] + '.npy')

                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3 :
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        if item[2] not in train_class: train_class.append(item[2])
                        train_label.append(item[2])
                        train_valid += 1

                if item[3] == 'test':
                    # video_dir = os.path.join('/data/users/xiaokang_peng/VGGsound/', 'test-videos/test-set-img', 'Image-{:02d}-FPS'.format(self.args.fps), item[0]+'_'+item[1]+'.mp4')
                    # audio_dir = os.path.join('/data/users/xiaokang_peng/VGGsound/', 'test-audios/test-set', item[0]+'_'+item[1]+'.wav')
                    # video_dir = os.path.join(dataroot, 'test/visual_all/visual', item[0]+'_'+item[1])
                    # audio_dir = os.path.join(dataroot, 'test/audio_np', item[0]+'_'+item[1] + '.npy')

                    video_dir = os.path.join(video_test_root, item[0]+'_'+item[1])
                    # audio_dir = os.path.join(audio_test_root, item[0]+'_'+item[1] + '.wav')
                    audio_dir = os.path.join(audio_test_root, item[0]+'_'+item[1] + '.npy')

                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3:
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        if item[2] not in test_class: test_class.append(item[2])
                        test_label.append(item[2])
                        test_valid += 1
        
        print("Get Valid Train Sample: " + str(train_valid))
        print("Get Valid Test Sample: " + str(test_valid))

        # print(len(train_class))
        # print(len(test_class))
        assert len(train_class) == len(test_class)
        # assert len(train_class) > 0
        if len(train_class) == 0:
            raise ValueError("If you see this, it means you have problem in reading dataset")

        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if self.mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
        else:
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]
        # image_samples = [os.listdir(self.video[idx])for idx in range(len(self.video))]
        # self.images = [[Image.open(os.path.join(self.video[idx], image_samples[idx][i])).convert('RGB') for i in range(self.args['use_video_frames'])]for idx in range(len(self.video))]
        # img = Image.open(os.path.join(self.video[idx], image_samples[i])).convert('RGB')
        # img = transform(img)
        # images[i] = img
        # from tqdm import tqdm
        # self.images = []
        # for idx in tqdm(range(len(self.video)), desc="Processing videos"):
        #     video_frames = []
        #     # 内层进度条显示帧处理进度（可选）
        #     for i in range(self.args['use_video_frames']):
        #         image_path = os.path.join(self.video[idx], image_samples[idx][i])
        #         video_frames.append(Image.open(image_path).convert('RGB'))
        #     self.images.append(video_frames)

    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx):
        # print(self.video[idx])
        # print(self.audio[idx])
        # audio
        # sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)
        # while len(sample)/rate < 10.:
        #     sample = np.tile(sample, 2)

        # start_point = random.randint(a=0, b=rate*5)
        # new_sample = sample[start_point:start_point+rate*5]
        # new_sample[new_sample > 1.] = 1.
        # new_sample[new_sample < -1.] = -1.

        # spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        # spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        
        spectrogram = np.load(self.audio[idx])

        # Def Image Transform
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        # for i in range(pick_num):
        #     if self.data_trans_mode == 'train':
        #         t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
        #         if t[i] >= 10:
        #             t[i] = 9
        #     else:
        #         t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1

        #     path1.append('frame_0000' + str(t[i]) + '.jpg')
        #     image.append(Image.open(path + "/" + path1[i]).convert('RGB'))  # 0-255
        #     image_arr.append(transf(image[i]))
        #     image_arr[i] = image_arr[i].unsqueeze(1).float()
        pick_num = self.args['use_video_frames']
        image_samples = os.listdir(self.video[idx])
        image_samples = sorted(image_samples)
        file_num = len(image_samples)
        select_index = np.random.choice(len(image_samples), size=pick_num, replace=False)
        select_index.sort()
        images = torch.zeros((pick_num, 3, 224, 224))
        t = [0] * pick_num
        seg = (file_num//pick_num)
        for i in range(pick_num):
            if self.mode == 'train':
                t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                if t[i] >= 10:
                    t[i] = 9
            else:
                t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1
        for i, idx_frame in enumerate(select_index):
            img_path = os.path.join(self.video[idx], image_samples[t[i]-1])
                # img_path = os.path.join(self.video[idx], image_samples[min(i * seg + max(int(seg/2), 1), len(image_samples)-1)])
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images[i] = img
        # images = transform(self.images[idx])

        spectrogram = torch.tensor(spectrogram).unsqueeze(0).float()
        images = images.permute(1,0,2,3)
        
        # label
        label = self.label[idx]

        # print(spectrogram.shape)

        return {
            'audio': spectrogram, 
            'visual': images, 
            'label': label,
            'idx': idx
        }


# import h5py
# in this situation, data is saved as h5py form by 1fps-preprocessing
# class VGGSound_h5py(Dataset)