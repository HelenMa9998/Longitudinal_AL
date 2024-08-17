import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from monai import transforms
from seed import setup_seed
import torch
import albumentations as A
import bisect


setup_seed()
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class MSSEG_Handler_2d(Dataset):
    def __init__(self, t0_image, t1_image, label, diff, mode="train"):
        self.t0_image = t0_image
        self.t1_image = t1_image
        self.diff = diff
        self.label = label
        
        if mode=="train":
            self.transform = A.Compose([
                A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, always_apply=False, p=0.5),
                A.Flip(p=0.5),
                A.Rotate(limit=90, interpolation=1,always_apply=False, p=0.5),
#                 A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
#                 A.Resize(336,336, interpolation=3, always_apply=True, p=1),
#                 A.ToTensor(),
        ], additional_targets={
            'image1': 'image',  # 第二个图像使用与第一个图像相同的处理
            'image2': 'image',
            'mask': 'mask'      # 标签
        })
        
        else:
            self.transform=None
                
    def __len__(self):
        return len(self.t0_image)

    def __getitem__(self, index):
        t0_image = self.t0_image[index].astype(np.float32)
        t1_image = self.t1_image[index].astype(np.float32)
        diff = self.diff[index].astype(np.float32)
        label = self.label[index].astype(np.uint8)
        if self.transform!=None: 
            transformed = self.transform(image=t0_image, image1=t1_image, image2=diff, mask=label)
            t0_image = transformed['image']
            t1_image = transformed['image1']
            diff = transformed['image2']
            label = transformed['mask']

        t0_image = torch.tensor(t0_image)
        t1_image = torch.tensor(t1_image)
        diff = torch.tensor(diff)
        label = torch.tensor(label)
        return t0_image, t1_image, diff, label, index