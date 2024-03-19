from PIL import Image  # Python Imaging Library
import cv2  # an image and video processing library
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ClassifyCountDataset(Dataset):
    def __init__(self, dataset_type, img_size, animal_stdevs: np.array = None):

        self.img_path = 'wild/JPEGImages/'
        self.classify_path = 'wild/class_annotations/'
        self.count_path = 'wild/count_annotations/'

        if animal_stdevs is None:
            self.animal_stdevs = np.ones(6)
        else:
            self.animal_stdevs = animal_stdevs

        # means and sds of images from ImageNet
        train_mean, train_sd = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_sd)
        ])
        self.files = pd.read_csv('wild/ImageSets/Main/{}.txt'.format(dataset_type), header=None)[0].tolist()

    def __getitem__(self, index):
        file = self.files[index]
        img = self.transforms(Image.fromarray(cv2.imread(
            self.img_path+file+'.jpg').astype(np.uint8)))
        classify_target = torch.from_numpy(
            np.load(self.classify_path+file+'.npy'))
        
        normed_count = np.load(self.count_path+file+'.npy') / self.animal_stdevs
        count_target = torch.from_numpy(normed_count)

        return img, classify_target, count_target

    def __len__(self):
        return len(self.files)
