import collections
import numpy as np
import os
import torch

from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CamvidDataset(Dataset):
    """Cam Video Dataset"""
    
    def __init__(
            self,
            root_dir=None,
            split=None,
            transform=None,
    ):
        """
        Args:
            root_dir(string): Root directory of the dataset
            split(string): Split being loaded in this dataset
            transform(torchvision.transforms): The transformation to be applied to image.        
        """

        self.num_classes = 12 
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.files = collections.defaultdict(list)
        if split:
            for split in ["train", "test", "val"]:
                files_list = os.listdir(os.path.join(root_dir, split))
                self.files[split] = files_list
        
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, idx):
        image_name = self.files[self.split][idx]
        image_path = os.path.join(self.root_dir, self.split, image_name)
        label_path = os.path.join(self.root_dir, self.split + "annot", image_name)
        
        image = io.imread(image_path)
        label = io.imread(label_path)
                
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return image, label

    def color_segmentation(self, labels):
        # Returns colored segmentation corresponding to the class labels.
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road = [128, 64, 128]
        Pavement = [60, 40, 222]
        Tree = [128, 128, 0]
        SignSymbol = [192, 128, 128]
        Fence = [64, 64, 128]
        Car = [64, 0, 128]
        Pedestrian = [64, 64, 0]
        Bicyclist = [0, 128, 192]
        Unlabelled = [0, 0, 0]
        
        label_colors = np.array([
            Sky, Building, Pole, Road, Pavement, Tree, SignSymbol,
            Fence, Car, Pedestrian, Bicyclist, Unlabelled
        ])

        colored_image = np.zeros((labels.shape[0], labels.shape[1], 3))
        for cls in range(0, self.num_classes):
            colored_image[labels == cls, :] = label_colors[cls, :]
        colored_image /= 255.0

        return colored_image
