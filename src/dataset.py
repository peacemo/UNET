'''
 # @ Author: Chen Xueqiang
 # @ Create Time: 2023-10-08 10:33:12
 # @ Modified by: Chen Xueqiang
 # @ Modified time: 2023-10-08 10:50:01
 # @ Description: The 'Carvana' dataset util. 
 '''


import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # 1. get the target image & mask file path (full, including filename and suffix)
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))

        # 2. load image & mask as np array (for augmentaion lib)
        image = np.array(Image.open(image_path).convert('RGB'))  # RGB by default
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)  # Greyscale
        mask[mask == 255.0] = 1.0  # convert all 255 -> 1, cuz we'r going to use a sigmoid function to activate the final prediction

        if self.transform is not None:  # apply transforms to the image and mask 
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask 