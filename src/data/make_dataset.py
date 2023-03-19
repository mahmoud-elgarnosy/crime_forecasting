import torch
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from skimage import io
from torchvision import transforms
from torch.utils.data import random_split
from src.data.backgroundSubtraction import BackgroundSubtraction
from torchvision.io import read_video, write_video
import numpy as np

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# we're just reading one image and its corresponding label
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        # self.classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'RoadAccident', 'Burglary', 'Explosion',
        #                 'Fighting', 'Robbery', 'Shooting', 'Stealing', 'Shoplifting', 'Vandalism', 'Normal']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        frames, _, _ = read_video(self.annotations.iloc[index, 0].split('..')[-1], pts_unit='sec')
        y_label = torch.tensor(0 if self.annotations.iloc[index, 1] == "Normal" else 1)
        i = 0
        while not frames.shape[0] == 301:
            frames, _, _ = read_video('..' + self.annotations.iloc[index + i, 0].split('..')[-1], pts_unit='sec')
            y_label = torch.tensor(0 if self.annotations.iloc[index + i, 1] == "Normal" else 1)

            i += 1
        bac_image = BackgroundSubtraction(frames[0], 5)

        # looping on videos to extract foreground
        for i, current_frame in enumerate(frames[4::5]):
            current_frame = np.array(current_frame)
            foreground_image = bac_image.subtract_background(current_frame)

            foreground_image = foreground_image[np.newaxis, ...]
            if i == 0:
                foreground_images = foreground_image
            else:
                foreground_images = np.vstack((foreground_images, foreground_image))

        # y_label = torch.tensor(self.classes.index(self.annotations.iloc[index, 1]))

        foreground_images = torch.tensor(foreground_images)
        foreground_images = foreground_images.permute(0, 3, 1,
                                                      2)  # keep dim 0 at dim0 and dim1 to be dim2 dnd dim2 to be dim1

        # if self.transform:
        #     image = self.transform(image)

        return foreground_images, y_label


if __name__ == '__main__':
    dataset = CustomDataset(csv_file='../../data/interim/train/train.csv', transform=transforms.ToTensor())

    train_set, test_set = random_split(dataset, [.8, .2])
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    videos, labels = next(iter(train_loader))
    print(videos.shape)
