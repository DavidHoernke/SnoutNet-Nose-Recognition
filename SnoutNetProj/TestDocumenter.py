import argparse

import torch
import numpy as np
import math
import csv
from torch.utils.data import DataLoader
from dataset import CustomImageDataset  # Assuming you already have this dataset
from model import SnoutNet  # Assuming you already have the SnoutNet model
from torchvision import transforms

# Helper function to calculate Euclidean distance
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

                # Calculate Euclidean distance
                dist = euclidean_distance(pred, label)
                distances.append(dist)

    # Convert distances list to numpy array for easier statistical calculations
    distances = np.array(distances)

    # Calculate statistics: min, mean, max, and standard deviation
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)

    return min_distance, mean_distance, max_distance, std_distance

if __name__ == "__main__":
    # Paths to annotations CSV and image folder
    annotations_dir = r'C:\Users\David Hoernke\PycharmProjects\SnoutNet-Nose-Recognition\SnoutNetProj\oxford-iiit-pet-noses\test_noses.txt'
    img_dir = r'C:\Users\David Hoernke\PycharmProjects\SnoutNet-Nose-Recognition\SnoutNetProj\oxford-iiit-pet-noses\images-original\images'  # Assuming your images are stored here

    print('running main ...')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('Using device:', device)

    # Default transforms with no augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL image to a tensor
    ])

    # Instantiate the CustomImageDataset for the test set
    test_dataset = CustomImageDataset(annotations_dir=annotations_dir, img_dir=img_dir, transform=transform)

    # Create a DataLoader for the test set
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # List of models to evaluate, skipping the ones you mentioned
    model_files = [f'{i}ColorAndBlurModel.pt' for i in range(50)]

    # CSV file to save the results
    results_file = 'model_evaluation_results.csv'

    # Write the header of the CSV
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Min Distance', 'Mean Distance', 'Max Distance', 'Std Distance'])

    # Loop through each model, load, evaluate, and save results
    for model_file in model_files:
        print(f'Evaluating {model_file}...')

        # Load the trained model
        model = SnoutNet()
        try:
            model.load_state_dict(torch.load(model_file))  # Load the saved model weights
        except:
            print("Failed to find model")
            continue
        model.to(device)

        # Evaluate the model on the test set
        min_distance, mean_distance, max_distance, std_distance = evaluate_model(model, test_loader, device)

        # Save the results to the CSV
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_file, min_distance, mean_distance, max_distance, std_distance])

    print(f"Model evaluations saved to {results_file}.")
