import argparse
import datetime

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchsummary import summary
from torchvision import transforms  # This is the correct import for transforms

from dataset import CustomImageDataset
from model import SnoutNet


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train(model, optimizer, criterion, train_loader, val_loader, scheduler, device, save_file="model.pt", plot_file='plot.png'):
    print('Training on device: {}', device)
    model.to(device)
    model.train()
    train_losses = []
    val_losses = []
    badLossCount = 0
    best_val_loss = float('inf')  # Initialize this outside the loop

    patience = 5  # Number of epochs to wait before stopping early

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        print(f"Epoch {epoch + 1}/{num_epochs} Starting, Learning rate:{optimizer.param_groups[0]['lr']}")

        # Iterate through batches of data
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device

            optimizer.zero_grad()  # Clear previous gradients
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()

            optimizer.step()

            # Track statistics
            running_loss += loss.item()

        # Calculate the average training loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)  # Track training loss

        #Validate the model on the validation set
        val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)  # Track validation loss

        # Adjust the learning rate based on validation loss
        scheduler.step(val_loss) #lets try it on training

        print(f'{datetime.datetime.now()} Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the best model and stop early if validation hasn't improved in `patience` steps
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            badLossCount = 0  # Reset the bad validation loss counter
            if save_file:
                torch.save(model.state_dict(), str(epoch)+save_file)
                print(f'Saving best model with validation loss: {val_loss:.4f}')
        else:
            badLossCount += 1  # Increment if validation loss didn't improve

        # Plotting after all epochs
        if plot_file is not None:
            plt.figure(figsize=(12, 14))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Time')
            plt.legend(loc='upper right')
            plt.grid(True)
            print(f'Saving loss plot to {plot_file}')
            plt.savefig(plot_file)

        if badLossCount >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    print("Training complete!")

def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accumulate validation loss
            val_loss += loss.item()

    # Calculate the average validation loss
    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')
    return val_loss

if __name__ == "__main__":
    #THESE ARE THE DEFAULT PARAMETERS OF THE MODEL *******************
    num_epochs = 50
    save_file = "model.pt"
    plot_file = "plot.png"
    batch_size = 32
    # END OF DEFAULT PARAMETERS OF THE MODEL ************************

    # Paths to annotations txt and image folder (TA if you are reading this please make sure these work for your proj :) )
    annotations_dir = './oxford-iiit-pet-noses/train_noses.txt'
    img_dir = './oxford-iiit-pet-noses/images-original/images'

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')

    args = argParser.parse_args()

    if args.s is not None:
        save_file = args.s
    if args.e is not None:
        num_epochs = args.e
    if args.b is not None:
        batch_size = args.b
    if args.p is not None:
        plot_file = args.p

    # Model Init.
    model = SnoutNet()
    model.to(device)
    model.apply(init_weights)
    summary(model, (3, 227, 227))

    # Transform with no augment
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL image to a tensor
    ])

    #Transforms including augmentations ( I just comment out the ones not being using )
    transformAugmented = transforms.Compose([
        # transforms.ColorJitter(brightness=(0.01,3), contrast=(0.01,3), saturation=(0.01,3)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor()
    ])

    #Transforms including augmentations ( I just comment out the ones not being using )
    transformAugmented2 = transforms.Compose([
        transforms.ColorJitter(brightness=(0.01,3), contrast=(0.01,3), saturation=(0.01,3)),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor()
    ])


    # Instantiate the CustomImageDataset
    dataset = CustomImageDataset(annotations_dir=annotations_dir, img_dir=img_dir, transform=transform)

    # Define split sizes (e.g., 80% train, 20% validation)
    dataset_size = len(dataset)

    print("Original dataset size: ", dataset_size)

    train_size = int(0.93 * dataset_size)
    val_size = dataset_size - train_size

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    datasetAugmented = CustomImageDataset(annotations_dir=annotations_dir, img_dir=img_dir, transform=transformAugmented)
    datasetAugmented2= CustomImageDataset(annotations_dir=annotations_dir, img_dir=img_dir, transform=transformAugmented2)

    train_dataset = ConcatDataset([train_dataset,datasetAugmented, datasetAugmented2])
    dataset.show_image_with_circle(567)
    datasetAugmented.show_image_with_circle(567)
    datasetAugmented2.show_image_with_circle(567) # some visualization of the same picture

    # Total size of the dataset
    print("Augmented Dataset Size:", len(train_dataset))

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00065)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Assuming `train_loader` and `val_loader` are your DataLoaders
    train(model,optimizer,criterion,train_loader,val_loader, scheduler, device, save_file, plot_file)
    # Iterate through the DataLoader (in your training or testing loop)
