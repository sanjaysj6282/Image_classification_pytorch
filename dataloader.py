from PIL import Image
from torch.utils.data import Dataset
import os
from glob import glob
import torch
import torchvision.transforms as transforms

# Dataset Class for Setting up the data loading process
# Stuff to fill in this script: _init_transform()
class inaturalist(Dataset):
    def __init__(self, root_dir, mode = 'train', transform = True):
        self.data_dir = root_dir
        self.mode = mode
        self.transforms = transform      
        self._init_dataset()
        if transform:
            self._init_transform()

    def _init_dataset(self):
        self.files = []
        self.labels = []
        dirs = sorted(os.listdir(os.path.join(self.data_dir, 'train')))
        if self.mode == 'train': 
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join(self.data_dir, 'train', dirs[dir], '*.jpg')))
                self.labels += [dir]*len(files)            
                self.files += files
        elif self.mode == 'val':
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join(self.data_dir, 'val', dirs[dir], '*.jpg')))
                self.labels += [dir]*len(files)            
                self.files += files
        else:
            print("No Such Dataset Mode")
            return None
    
    def _init_transform(self):
        self.transform = transforms.Compose([
            # Useful link for this part: https://pytorch.org/vision/stable/transforms.html
            #----------------YOUR CODE HERE---------------------#
        ])
    
    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        label = self.labels[index]

        if self.transforms:
            img = self.transform(img)

        label = torch.tensor(label, dtype = torch.long)

        return img, label

    def __len__(self):
        return len(self.files)