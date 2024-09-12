import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from PIL import Image

class DermnetDataset(Dataset):
    def __init__(self, imgs_cv,transform=None):
        self.imgs_cv = imgs_cv
        self.transform = transform
        
        classes = torch.tensor(imgs_cv['class'])
        self.labels = F.one_hot(classes)
        self.labels=torch.Tensor(self.labels).float()
        
    def __len__(self):
        return self.imgs_cv.shape[0]
        
    def __getitem__(self, idx):
        img = Image.open(self.imgs_cv.loc[idx, 'path'])
        img = img.resize((256, 256))
        if self.transform:
            img = self.transform(img)
            
        return img, self.labels[idx]