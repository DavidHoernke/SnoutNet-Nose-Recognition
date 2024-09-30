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


# Helper function to rescale the coordinates back to the original image size
def rescale_coordinates(normalized_coords, original_size):
    width, height = original_size
    return int(normalized_coords[0] * width), int(normalized_coords[1] * height)


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

            # Calculate Euclidean distances for each prediction in the batch
            for i in range(len(labels)):
                pred = outputs[i].cpu().numpy()  # Convert prediction to numpy array
                label = labels[i].cpu().numpy()  # Convert ground truth to numpy array
                dist = euclidean_distance(pred, label)  # Calculate the Euclidean distance
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
        # Get the image and label (normalized coordinates)
        img, label = test_dataset[idx]

        original_size = img.size  # Get the original image size

        img = img.to(device).unsqueeze(0)  # Add batch dimension and move to device

        with torch.no_grad():
            pred = model(img).cpu().numpy()[0]  # Get the predicted coordinates (normalized)

        # Rescale predicted and ground truth coordinates back to original size
        pred_x, pred_y = rescale_coordinates(pred, original_size)
        gt_x, gt_y = rescale_coordinates(label.cpu().numpy(), original_size)

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
    # Paths to annotations CSV and image folder
    annotations_dir = r'C:\Users\David Hoernke\PycharmProjects\SnoutNet-Nose-Recognition\SnoutNetProj\oxford-iiit-pet-noses\test_noses.txt'
    img_dir = r'C:\Users\David Hoernke\PycharmProjects\SnoutNet-Nose-Recognition\SnoutNetProj\oxford-iiit-pet-noses\images-original\images'  # Assuming your images are stored here

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('Using device:', device)

    # Load the trained model
    model = SnoutNet()
    model.load_state_dict(torch.load('model.pt'))  # Load your saved model weights
    model.to(device)

    # Define any transformations (optional)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL image to a tensor
    ])

    # Instantiate the CustomImageDataset for the test set
    test_dataset = CustomImageDataset(annotations_dir=annotations_dir, img_dir=img_dir, transform=transform)

    # Create a DataLoader for the test set
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, device)

    # Visualize predictions on 10 randomly selected images
    visualize_predictions(model, test_dataset, device, num_samples=10, seeRummy=True, rummy_image_name="rummy_1.jpg")
