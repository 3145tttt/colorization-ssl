import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image
import random

class FoodDataset(Dataset):
    
    def __init__(self, path, files, targets, transform, class_to_id, id_to_class):

        self.path = path
        self.transform = transform
        self.files = files
        self.targets = targets
        
        self.class_to_id, self.id_to_class = class_to_id, id_to_class
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(f"{self.path}/food-101/images/{file}.jpg").convert('RGB')
        img = self.transform(img)
        img = torch.clip(img, 0, 1)
        
        target = self.targets[idx]
        return img, target

class FoodColorizationDataset(FoodDataset):
    def __init__(self, path, files, targets, transform, class_to_id, id_to_class, augmentation_transform):
        super().__init__(path, files, targets, transform, class_to_id, id_to_class)
        
        self.transform2 = augmentation_transform
        
    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(f"{self.path}/food-101/images/{file}.jpg").convert('RGB')
        
        target = self.transform(img)
        img = self.transform(img)
        
        if random.random() > 0.5:
            target = F.hflip(target)
            img = F.hflip(img)
            
        img = self.transform2(img)
        
        img = F.normalize(img, 0.5, 0.5, False)
        target = F.normalize(target, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
        return img, target