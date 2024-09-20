import json
from PIL import Image
import random
from tqdm import tqdm
from pathlib import Path
import shutil
import os

# Function to save augmented images for each class in train/val/test datasets
def save_images(image_paths, class_name,  dataset_type, output_folder_path, transform):
    """
    Save the images for a specific dataset type (train/val/test) in the specified folder.
    Optionally, augment and save images.
    """

    # Create the destination folder if it doesn't exist
    destination_folder = output_folder_path / dataset_type / class_name
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Find padding for naming images
    num_images = len(image_paths)
    pad_len = len(str(num_images))
    
    if num_images == 0:
        return

    # Iterate through image paths, augment and save them
    for count, img_path in enumerate(image_paths):
        save_path = destination_folder / f"{class_name}_{count:0{pad_len}}.jpg"
        augment_image(img_path, save_path, transform)


def sample_and_augment(image_paths, class_name, dataset_type, output_folder_path, transform, N):

    # Create the destination folder if it doesn't exist
    destination_folder = output_folder_path / dataset_type / class_name
    destination_folder.mkdir(parents=True, exist_ok=True)
        
    # Find padding for naming images
    pad_len = len(str(N))
    num_images = len(image_paths)
    
    # No images so return
    if num_images == 0:
        return
    
    # Augment
    if num_images < N:
        transform = transform.train()
        
        sampled_images = random.choices(image_paths, k=N - num_images)
        sampled_images.extend(image_paths)
        for count, img_path in enumerate(sampled_images):
            save_path = f"{destination_folder}/{class_name}_{count:0{pad_len}}.jpg"
            augment_image(img_path, save_path, transform)
    
    # Downsample
    else:
        transform = transform.eval()
        sampled_images = random.sample(image_paths, N)
        for count, img_path in enumerate(sampled_images):
            save_path = f"{destination_folder}/{class_name}_{count:0{pad_len}}.jpg"
            augment_image(img_path, save_path, transform)


def augment_image(img_path, save_path, transform):
    """
    Apply augmentation to an image and save the result.
    """
    image = Image.open(img_path)    # Open image
    image = transform(image)        #  Apply augmentation
    image.save(save_path)
    
# Function to create train/val/test datasets with optional oversampling for training
def create_dataset_splits(splits_data, img_folder_path, output_folder_name, transform, dataset_type='train', N=None):
    """
    Create a dataset using the provided splits data (train, valid, test) with optional oversampling.
    Saves augmented images in the appropriate folders.
    """
    # Define where the images will be saved
    img_dir = img_folder_path.parent
    output_folder_path = img_dir / output_folder_name
    
    for class_id, image_data in tqdm(splits_data['images'].items()):
        class_name = splits_data['categories'][class_id]

        # Get all image paths for this class and dataset type
        image_paths = [img_folder_path / class_name / img for img in image_data[dataset_type]]

        
        # If training then augment to make sure even number of images per class
        if dataset_type == "train" and N:
            sample_and_augment(image_paths, class_name, dataset_type, output_folder_path, transform, N)
            
        else:
            transform = transform.eval()
            save_images(image_paths, class_name, dataset_type, output_folder_path, transform)


def load_json(filepath):
    """
    Load JSON data from the provided file path.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# Define the split_datasets function
def split_datasets(dataset_name, transform, CLASS_SPLIT_PATHS, IMG_FOLDER_PATHS, OUTPUT_FOLDER_NAMES, N=None):
    """
    Load the train, validation, and test datasets with optional oversampling for training.
    """
    class_split_path = Path(CLASS_SPLIT_PATHS.get(dataset_name))
    output_folder_name = Path(OUTPUT_FOLDER_NAMES.get(dataset_name))
    img_folder_path = Path(IMG_FOLDER_PATHS.get(dataset_name)) 
    
    if class_split_path is None:
        raise KeyError(f'{dataset_name} not in available datasets: 'f'{", ".join(CLASS_SPLIT_PATHS.keys())}')
    
    # Load the splits.json file containing partitions and categories
    class_split_path = class_split_path / "splits.json"
    splits_data = load_json(class_split_path)

    # Create datasets for train, validation, and test splits
    print("Augmenting and resizing training images...")
    create_dataset_splits(splits_data, img_folder_path, output_folder_name, transform, dataset_type="train", N=N)
    
    print("Resizing validation images...")
    create_dataset_splits(splits_data, img_folder_path, output_folder_name, transform, dataset_type="valid")
    
    print("Resizing testing images...")
    create_dataset_splits(splits_data, img_folder_path, output_folder_name, transform, dataset_type="test")

    
