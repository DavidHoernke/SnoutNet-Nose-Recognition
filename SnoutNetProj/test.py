import argparse

import torch
import numpy as np
import math
import random

from torch.utils.data import DataLoader
from dataset import CustomImageDataset  # Assuming you already have this dataset
from model import SnoutNet  # Assuming you already have the SnoutNet model
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image


# Helper function to calculate Euclidean distance directly in pixel space
def euclidean_distance(prediction, ground_truth):
    """Calculate the Euclidean distance between predicted and ground truth coordinates."""
    return math.sqrt((prediction[0] - ground_truth[0]) ** 2 + (prediction[1] - ground_truth[1]) ** 2)


def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    distances = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass to get predictions
            outputs = model(images)

            for i in range(len(labels)):
                pred = outputs[i].cpu().numpy()  # Get predicted coordinates in pixel space
                label = labels[i].cpu().numpy()  # Get ground truth coordinates in pixel space

                # Calculate Euclidean distance in pixel space
                dist = euclidean_distance(pred, label)
                distances.append(dist)

    # Convert distances list to numpy array for easier statistical calculations
    distances = np.array(distances)

    # Calculate statistics: min, mean, max, and standard deviation
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)

    # Print the results
    print(f'Min Euclidean Distance: {min_distance:.4f}')
    print(f'Mean Euclidean Distance: {mean_distance:.4f}')
    print(f'Max Euclidean Distance: {max_distance:.4f}')
    print(f'Standard Deviation of Euclidean Distance: {std_distance:.4f}')

    return min_distance, mean_distance, max_distance, std_distance


def visualize_predictions(model, test_dataset, device, num_samples=10, seeRummy=True, rummy_image_name="rummy_1.jpg"):
    """
    Randomly select num_samples images from the test dataset and visualize the predictions.

    Parameters:
    - model: The trained model.
    - test_dataset: The dataset to sample images from.
    - device: 'cpu' or 'cuda'.
    - num_samples: Number of random images to display (excluding Rummy).
    - seeRummy: Whether to include Rummy's image.
    - rummy_image_name: The file name of Rummy's image.
    """
    model.eval()

    # Find Rummy's image index based on the file name
    rummy_index = None
    for i in range(len(test_dataset)):
        img_name = test_dataset.img_labels.iloc[i, 0]  # Assuming file name is in the first column
        if rummy_image_name in img_name:
            rummy_index = i
            break

    if rummy_index is None:
        print(f"Rummy's image '{rummy_image_name}' not found in the dataset.")
        return

    # Randomly select num_samples images, excluding Rummy's image (to avoid duplication)
    indices = random.sample([i for i in range(len(test_dataset)) if i != rummy_index], num_samples)

    # If seeRummy is True, include Rummy's image
    if seeRummy:
        indices.pop()  # Remove one random image to make space for Rummy
        indices.append(rummy_index)  # Ensure Rummy's image is included

    # Create a figure for plotting
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 images per row

    for ax, idx in zip(axes.flatten(), indices):
        # Get the image and label (coordinates in pixel space)
        img, label = test_dataset[idx]

        img = img.to(device).unsqueeze(0)  # Add batch dimension and move to device

        with torch.no_grad():
            pred = model(img).cpu().numpy()[0]  # Get the predicted coordinates in pixel space

        # Ground truth coordinates in pixel space
        gt_x, gt_y = label.cpu().numpy()

        # Predicted coordinates in pixel space
        pred_x, pred_y = pred

        # Convert the image to PIL for drawing
        pil_img = transforms.ToPILImage()(img.squeeze().cpu())  # Convert back to PIL

        # Draw the predicted and ground truth coordinates
        draw = ImageDraw.Draw(pil_img)
        radius = 5

        # Draw ground truth (green)
        draw.ellipse((gt_x - radius, gt_y - radius, gt_x + radius, gt_y + radius), outline="green", width=2)

        # Draw predicted (red)
        draw.ellipse((pred_x - radius, pred_y - radius, pred_x + radius, pred_y + radius), outline="red", width=2)

        # Show the image with the predictions
        ax.imshow(pil_img)
        ax.set_title(f'GT (green), Pred (red)')
        ax.axis('off')  # Hide axis

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Paths to annotations txt and image folder (TA if you are reading this please make sure these work for your proj :) )
    annotations_dir = './oxford-iiit-pet-noses/test_noses.txt'
    img_dir = './oxford-iiit-pet-noses/images-original/images'

    modelName = "StandardModel.pt"

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='ModelName', type=str, help='Name of model (.pt)')

    args = argParser.parse_args()

    if args.m is not None:
        modelName = args.m


    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('Using device:', device)

    # Load the trained model
    model = SnoutNet()
    model.load_state_dict(torch.load(modelName))  # Load your saved model weights
    model.to(device)

    # Define any transformations (optional)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL image to a tensor
    ])

    # Instantiate the CustomImageDataset for the test set
    test_dataset = CustomImageDataset(annotations_dir=annotations_dir, img_dir=img_dir, transform=transform)

    # Create a DataLoader for the test set
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, device)

    # Visualize predictions on 10 randomly selected images
    # I also include a picture of my roommates dog, Rummy, for experimentation on a new animal
    visualize_predictions(model, test_dataset, device, num_samples=10, seeRummy=True, rummy_image_name="rummy_1.jpg")
