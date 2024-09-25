from dataset import CustomImageDataset  # Import the CustomImageDataset class

def showMeACutePetAndWhereTheirNoseIs(index:int):
    # Paths to annotations CSV and image folder
    annotations_dir = r'C:\Users\David Hoernke\PycharmProjects\SnoutNet-Nose-Recognition\SnoutNetProj\oxford-iiit-pet-noses\train_noses.txt'
    img_dir = r'C:\Users\David Hoernke\PycharmProjects\SnoutNet-Nose-Recognition\SnoutNetProj\oxford-iiit-pet-noses\images-original\images'  # Assuming your images are stored here

    # Instantiate the dataset
    dataset = CustomImageDataset(annotations_dir=annotations_dir, img_dir=img_dir)

    # Show the first image with a circle around the coordinates
    dataset.show_image_with_circle(index, radius=100, color="red")