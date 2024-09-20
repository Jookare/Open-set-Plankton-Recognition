import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import json

class PlanktonSet(Dataset):
    def __init__(self, path, known_classes, classes, mapping, transform, gallery = False):

        self.transform = transform
        self.mapping = mapping

        if gallery:
            self.images, self.labels = get_images(path, classes, start = 0, end = 100)
        else:
            self.images, self.labels = get_images(path, classes)
        
        # Get the class names from mapping
        self.class_names = [mapping[class_id] for class_id in set(self.labels)]

        # Find the unknown class names
        self.unknown_classes = sorted(list(set(self.class_names) - set(known_classes)))

        # Find the indices of unknown classes in self.class_names
        self.unknown_class_id = [i for i, class_name in enumerate(self.class_names) if class_name in self.unknown_classes]
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()
        image_path = self.images[idx]

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        return image, label

 
def get_images(main_folder_path, classes, start=None, end=None):
    path = main_folder_path
    images = []
    labels = []

    # Loop through each subfolder
    for idx, class_name in enumerate(classes):
        image_files = get_class_images(path, class_name)

        if (start is not None and end is not None ):
            labels += [idx] * (end - start)
            images += image_files[start:end]
        else:
            labels += [idx] * len(image_files)
            images += image_files

    return images, labels

def get_class_images(main_folder_path, class_name):
    folder_path = main_folder_path / class_name
    # List all image files
    image_files = [str(f) for f in folder_path.glob("*") if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    return image_files


def load_json(filepath):
    """
    Load JSON data from the provided file path.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def map_class_indices_to_names(trial_classes, categories):
    # Convert the category keys to integers for easier comparison
    categories = {int(k): v for k, v in categories.items()}
    
    # Map known and unknown indices to class names
    known_class_names = [categories[idx] for idx in trial_classes['Known']]
    unknown_class_names = [categories[idx] for idx in trial_classes['Unknown']]
    
    new_class_mapping = {}
    
    # Assign new indices to known classes (0 to len(known) - 1)
    for new_idx, class_name in enumerate(known_class_names):
        new_class_mapping[new_idx] = class_name
    
    # Assign new indices to unknown classes (len(known) to len(known) + len(unknown) - 1)
    for new_idx, class_name in enumerate(unknown_class_names, start=len(known_class_names)):
        new_class_mapping[new_idx] = class_name
    
    return {
        'Known': known_class_names,
        'Unknown': unknown_class_names,
        'Mapping': new_class_mapping
    }



def get_datasets(data_dir, trial_path, trial, transform, unk_in_valid = False, gallery = False):
    # Define the paths for datasets
    train_path = Path(data_dir) / "train"
    valid_path = Path(data_dir) / "valid"
    test_path = Path(data_dir) / "test"
    
    # Load the trial and categories information from jsons
    trial_classes = load_json(trial_path / f"{trial}.json")
    splits_data = load_json(trial_path / "splits.json")
    
    # Find the known and unknown class names and mapping from id to class name
    mapped_classes = map_class_indices_to_names(trial_classes, splits_data["categories"])
    
    # Define train, test and validation classes
    train_classes = mapped_classes['Known']
    test_classes = mapped_classes['Known'] + mapped_classes['Unknown']
    mapping = mapped_classes['Mapping']
    if unk_in_valid:
        valid_classes = test_classes
    else:
        valid_classes = train_classes
    
    # Define the transforms
    if isinstance(transform, tuple):
        transform_length = len(transform)
        if transform_length == 2:
            transform_train, transform_valid = transform
            transform_test = transform_valid
        elif transform_length == 3:
            transform_train, transform_valid, transform_test = transform
        else:
            raise ValueError('Too many transformations in the input.')
    else:
        transform_train = transform
        transform_valid = transform
        transform_test  = transform
    
    # Get the datasets
    train_dataset = PlanktonSet(train_path, mapped_classes['Known'], train_classes, mapping, transform=transform_train, gallery=gallery)
    valid_dataset = PlanktonSet(valid_path, mapped_classes['Known'], valid_classes, mapping, transform=transform_valid)
    test_dataset = PlanktonSet(test_path, mapped_classes['Known'], test_classes, mapping, transform=transform_test)
    
    return (train_dataset, valid_dataset, test_dataset)
    