import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from torchvision import transforms  # This is the correct import for transforms

from dataset import CustomImageDataset
from model import SnoutNet


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_file=None, plot_file=None):
    print('training ...')
    model.train()


    for epoch in range(num_epochs):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Optionally validate the model and adjust learning rate based on validation loss
        val_loss = validate(model, val_loader, criterion, device)

        # Step the scheduler based on the validation loss (for ReduceLROnPlateau)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        scheduler.step(loss_train) # work in progress here
        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))

        if save_file != None:
            torch.save(model.state_dict(), save_file)

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

if __name__ == "__main__":
    # Paths to annotations CSV and image folder
    annotations_dir = r'C:\Users\David Hoernke\PycharmProjects\SnoutNet-Nose-Recognition\SnoutNetProj\oxford-iiit-pet-noses\train_noses.txt'
    img_dir = r'C:\Users\David Hoernke\PycharmProjects\SnoutNet-Nose-Recognition\SnoutNetProj\oxford-iiit-pet-noses\images-original\images'  # Assuming your images are stored here

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    model = SnoutNet()
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)
    # Define any transformations (optional)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL image to a tensor
        # transforms.Normalize((0.5,), (0.5,))  # Normalization (if required)
    ])

    # Instantiate the CustomImageDataset
    dataset = CustomImageDataset(annotations_dir=annotations_dir, img_dir=img_dir, transform=transform)

    # Total size of the dataset
    dataset_size = len(dataset)

    # Define split sizes (e.g., 80% train, 20% validation)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set number of epochs
    num_epochs = 10

    # Assuming `train_loader` and `val_loader` are your DataLoaders
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    # Iterate through the DataLoader (in your training or testing loop)
    for batch_idx, (images, labels) in enumerate(data_loader):
        # Now you have a batch of images and corresponding labels
        print(images.shape)  # e.g., torch.Size([32, 3, 227, 227]) for batch of 32 RGB images
        print(labels)
