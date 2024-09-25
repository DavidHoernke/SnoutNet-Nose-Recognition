import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


    def show_image_with_circle(self, idx, radius=10, color="red"):
        # Get the image and coordinates
        img, coordinates = self.__getitem__(idx)

        # If the image is a tensor, convert it back to a PIL Image
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # Draw the red circle around the coordinates
        draw = ImageDraw.Draw(img)

        coordinates = eval(coordinates)

        x, y = coordinates
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)

        # Show the image
        plt.imshow(img)
        plt.axis('off')  # Hide axis
        plt.show()