from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
import torch
import sys

sys.path.append('./')
sys.path.append('../')
from trainer import Trainer
from dataset_process import dataset_creator
from train_utils import train_test_transform, get_device, get_dataset_path
from backbones import Backbone

def parse_arguments():
    parser = argparse.ArgumentParser(description='Open Set Classifier Training')
    parser.add_argument('--trial', required=False, type=int, help='Trial number, 0-4 provided')
    parser.add_argument('--backbone', default=None, type=str, help='Define backbone model', choices=['resnet18', 'densenet121'])
    parser.add_argument('--num_unk', required=True, type=int, help='Number of unknown classes. Set 0 to train in closed set mode')
    parser.add_argument('--no_normalize', action='store_false', help="If set, normalization will be disabled")
    parser.add_argument('--dataset', required=True, type=str, help='Name of the dataset', choices=['zooplankton', 'phytoplankton'])
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--name', default="", type=str, help='Optional name for saving')
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = get_device()
    path = get_dataset_path(args.dataset)

    if path is None:
        raise ValueError(f"Invalid dataset name: {args.dataset}")
    print(f"Dataset path: {path}")

    # Data transforms
    transform_train = train_test_transform(num_channels=3, normalize=args.no_normalize).train()
    transform_test = train_test_transform(num_channels=3, normalize=args.no_normalize).eval()
    transform = (transform_train, transform_test)

    # Define the unknown indices
    start_idx = args.num_unk * args.trial
    unk_idx = list(range(start_idx, start_idx + args.num_unk))

    # Define datasets
    batch_size = args.batch_size
    train_dataset, val_dataset, _ = dataset_creator.prepare_datasets(path, transform, unk_idx=unk_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=10)
    num_classes = train_dataset.num_classes

    # Model
    model = Backbone(args.backbone, num_classes)
    model = model.to(device)

    # Define optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-6)

    # Train the model
    trainer = Trainer(model, train_loader, valid_loader, loss_fn, optimizer, device)
    trainer.fit(n_epochs=100, save_name=f"OpenMax_{args.backbone}_{args.trial}_{args.name}")


if __name__ == "__main__":
    main()