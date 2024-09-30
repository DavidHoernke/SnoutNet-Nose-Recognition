import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

TARGET_SIZE = (227, 227)


# Helper function for image and coordinate rescaling
def rescale_image_and_coords(image, coordinates):
    """Rescale image to target size and scale the coordinates accordingly."""
    # Get original image size
    original_width, original_height = image.size

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize the image
    image_resized = image.resize(TARGET_SIZE)

    # Calculate the scaling factors for width and height
    width_scale = TARGET_SIZE[0] / original_width
    height_scale = TARGET_SIZE[1] / original_height

    # Scale the coordinates
    x, y = coordinates
    x_normalized = x / original_width
    y_normalized = y / original_height

    return image_resized, torch.tensor((x_normalized, y_normalized), dtype=torch.float)


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

        try:
            img = Image.open(img_path)  # Use PIL Image directly
        except:
            print(f"Corrupted image detected: {img_path}. Replacing with a random image.")

            Idx = random.randint(0, len(self.img_labels))
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[Idx, 0])
            img = Image.open(img_path)

        label = eval(self.img_labels.iloc[idx, 1])  # Read the string '(145, 293)' as a tuple

        img, label = rescale_image_and_coords(img, label)

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
        x, y = coordinates  # Now coordinates is a tuple of two integers
        draw = ImageDraw.Draw(img)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)

        # Show the image
        plt.imshow(img)
        plt.axis('off')  # Hide axis
        plt.show()
