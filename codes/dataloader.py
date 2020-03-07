from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import simulation

import os

class Pick_images(Dataset):
    def __init__(self,data_dir, transform=None):
        self.input_images = datasets.ImageFolder(os.path.join(data_dir,"imgs"))
        self.target_masks = datasets.ImageFolder(os.path.join(data_dir,"masks"))     
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):  
        image = self.input_images[idx][0]

        mask = self.target_masks[idx][0]
        if self.transform:
            image = self.transform(image)
            mask = transforms.Compose([transforms.CenterCrop((200,300)),transforms.ToTensor(),])(mask)



        
        return image, mask


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]