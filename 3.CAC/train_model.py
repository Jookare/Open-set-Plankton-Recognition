from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
import torch
import sys

sys.path.append('./')
sys.path.append('../')
from utils import CACLoss
from trainer import Trainer
from dataset_process import dataset_creator
from train_utils import train_test_transform, get_device, get_dataset_path, get_trial_path
from backbones import Backbone


def parse_arguments():
    parser = argparse.ArgumentParser(description='Open Set Classifier Training')
    parser.add_argument('--trial', required = False, type = int, help='Trial number, 0-4 provided')
    parser.add_argument('--backbone', default = None, type = str, help='Define backbone model', choices = ['resnet18', 'densenet121'])
    parser.add_argument('--magnitude', default = 10, type = int, help='Magnitude of the anchor point')
    parser.add_argument('--scale', default = 0.1, type = float, help='Weighting of Anchor loss component')
    parser.add_argument('--no_normalize', action='store_false', help="default to True, if set on then false")
    parser.add_argument('--dataset', required=True, type = str, help='Name of the model file', choices = ['zooplankton', 'phytoplankton']) 
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--name', default="", type = str, help='Optional name for saving') 
    return parser.parse_args()
    

def main():
    args = parse_arguments()
    device = get_device()
    path = get_dataset_path(args.dataset)

    assert path is not None, f"Invalid dataset name: {path}"

    # Data transforms
    transform_train = train_test_transform(num_channels=3, normalize=args.no_normalize).train()
    transform_test = train_test_transform(num_channels=3, normalize=args.no_normalize).eval()
    transform = (transform_train, transform_test)

    # Define parameters for datasets
    batch_size = args.batch_size
    trial_path = get_trial_path(args.dataset)
    
    # Define datasets
    train_dataset, valid_dataset, _ = dataset_creator.get_datasets(path, trial_path, args.trial, transform, unk_in_valid=False, gallery=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    num_classes = train_dataset.num_classes

    # Initialize the backbone
    model = Backbone(args.backbone, num_classes)
    model = model.to(device)

    # Define optimizer and loss function
    loss_fn = CACLoss(num_classes, args.magnitude, args.scale, device=device)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-6)

    # Train the model
    trainer = Trainer(model, train_loader, valid_loader, loss_fn, optimizer, device)
    trainer.fit(n_epochs=100, save_name=f"{args.backbone}_cac_{args.trial}_{args.name}")

if __name__ == "__main__":
    main()