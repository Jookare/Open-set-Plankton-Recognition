import torch
from torchvision.transforms import v2
import torch.nn as nn
from pathlib import Path

class train_test_transform(nn.Module):
    
    def __init__(self, num_channels, grayscale=True, normalize=True):
        super(train_test_transform, self).__init__()
        self.num_channels = num_channels
        self.normalize = normalize
        self.grayscale = grayscale

    def forward(self, img):

        # Define transforms
        transform_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]

        if self.grayscale:
            transform_list.insert(
                0, v2.Grayscale(num_output_channels=self.num_channels),
            )

        # If the transform is used in training then set a random affine as the third element
        if self.training:
            transform_list.insert(
                1, v2.RandomAffine(degrees=(0, 180), translate=(0, 0.1), scale=(0.95, 1.05), fill=255)
            )

        # If normalization is required
        if self.normalize:
            transform_list.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        # Compose the transform
        transform = v2.Compose(transform_list)

        return transform(img)

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return device


def get_dataset_path(dataset):
    dataset_paths = {
        "zooplankton": "../Data/images/zooplankton_224",
        "phytoplankton": "../Data/images/phytoplankton_224"
    }
    return dataset_paths.get(dataset)

def get_trial_path(dataset):
    trial_paths = {
        "zooplankton": "../data/class_splits/SYKE-plankton_ZooScan_2024",
        "phytoplankton": "../data/class_splits/SYKE-plankton_IFCB_2022"
    }
    return Path(trial_paths.get(dataset))
